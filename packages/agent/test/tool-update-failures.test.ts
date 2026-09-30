import { setImmediate } from "node:timers/promises";
import { type AssistantMessage, createAssistantMessageEventStream, type Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { runAgentLoop, runToolCall } from "../src/agent-loop.ts";
import type { AgentEvent, AgentTool, AgentToolCall, AgentToolResult, ToolExecutionMode } from "../src/types.ts";

const toolCall: AgentToolCall = { type: "toolCall", id: "call", name: "progress", arguments: {} };
const assistantMessage: AssistantMessage = {
	role: "assistant",
	content: [toolCall],
	api: "openai-responses",
	provider: "openai",
	model: "mock",
	usage: {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	},
	stopReason: "toolUse",
	timestamp: 0,
};
const result: AgentToolResult = { content: [{ type: "text", text: "progress" }], details: {} };

function progressTool(execute: AgentTool["execute"]): AgentTool {
	return {
		name: "progress",
		label: "Progress",
		description: "Reports progress",
		parameters: Type.Object({}),
		execute,
	};
}

describe("tool update failures", () => {
	it.each(["sync", "async"])("returns an error outcome when an update callback fails (%s)", async (mode) => {
		const tool = progressTool(async (_id, _params, _signal, onUpdate) => {
			onUpdate?.(result);
			return result;
		});
		const afterCalls: boolean[] = [];
		await expect(
			runToolCall(toolCall, {
				tools: [tool],
				assistantMessage,
				context: { messages: [] },
				onUpdate: () => {
					if (mode === "sync") throw new Error("update failed");
					return Promise.reject(new Error("update failed"));
				},
				afterToolCall: async ({ isError }) => {
					afterCalls.push(isError);
					return undefined;
				},
			}),
		).resolves.toMatchObject({
			isError: true,
			result: { content: [{ type: "text", text: "update failed" }] },
		});
		expect(afterCalls).toEqual([true]);
	});

	it("handles rejected updates while the tool is still running", async () => {
		const tool = progressTool(async (_id, _params, _signal, onUpdate) => {
			onUpdate?.(result);
			await setImmediate();
			return result;
		});
		await expect(
			runToolCall(toolCall, {
				tools: [tool],
				assistantMessage,
				context: { messages: [] },
				onUpdate: () => Promise.reject(new Error("early update failed")),
			}),
		).resolves.toMatchObject({
			isError: true,
			result: { content: [{ type: "text", text: "early update failed" }] },
		});
	});

	it.each([false, true])("waits for all updates after a failure (tool also fails: %s)", async (toolFails) => {
		const pendingUpdate = Promise.withResolvers<void>();
		const tool = progressTool(async (_id, _params, _signal, onUpdate) => {
			onUpdate?.(result);
			onUpdate?.(result);
			if (toolFails) throw new Error("tool failed");
			return result;
		});
		let updates = 0;
		let settled = false;
		const outcome = runToolCall(toolCall, {
			tools: [tool],
			assistantMessage,
			context: { messages: [] },
			onUpdate: () => (++updates === 1 ? Promise.reject(new Error("update failed")) : pendingUpdate.promise),
		}).then(
			(value) => {
				settled = true;
				return { value };
			},
			(error: unknown) => {
				settled = true;
				return { error };
			},
		);
		try {
			await setImmediate();
			expect(settled).toBe(false);
		} finally {
			pendingUpdate.resolve();
			await outcome;
		}
		expect(await outcome).toMatchObject({
			value: {
				isError: true,
				result: { content: [{ type: "text", text: toolFails ? "tool failed" : "update failed" }] },
			},
		});
	});

	it.each<ToolExecutionMode>(["sequential", "parallel"])(
		"completes the tool lifecycle and continues the loop after a rejected update (%s)",
		async (toolExecution) => {
			const tool = progressTool(async (_id, _params, _signal, onUpdate) => {
				onUpdate?.(result);
				await setImmediate();
				return result;
			});
			const model: Model<"openai-responses"> = {
				id: "mock",
				name: "Mock",
				api: "openai-responses",
				provider: "openai",
				baseUrl: "https://example.invalid",
				reasoning: false,
				input: ["text"],
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: 8192,
				maxTokens: 2048,
			};
			const events: AgentEvent[] = [];
			let requests = 0;
			const messages = await runAgentLoop(
				[{ role: "user", content: "Run the tool", timestamp: 0 }],
				{ messages: [], tools: [tool] },
				{
					model,
					toolExecution,
					convertToLlm: (messages) =>
						messages.filter(
							(message) =>
								message.role === "system" ||
								message.role === "user" ||
								message.role === "assistant" ||
								message.role === "toolResult",
						),
				},
				async (event) => {
					events.push(event);
					if (event.type === "tool_execution_update") throw new Error("listener failed");
				},
				undefined,
				() => {
					const stream = createAssistantMessageEventStream();
					const reason = requests++ === 0 ? "toolUse" : "stop";
					const message: AssistantMessage = {
						...assistantMessage,
						content: reason === "toolUse" ? [toolCall] : [{ type: "text", text: "Recovered" }],
						stopReason: reason,
					};
					stream.push({ type: "done", reason, message });
					return stream;
				},
			);
			expect(messages.find((message) => message.role === "toolResult")).toMatchObject({
				toolCallId: "call",
				isError: true,
				content: [{ type: "text", text: "listener failed" }],
			});
			expect(events.find((event) => event.type === "tool_execution_end")).toMatchObject({
				toolCallId: "call",
				isError: true,
			});
			expect(events.at(-1)?.type).toBe("agent_end");
			expect(requests).toBe(2);
		},
	);
});
