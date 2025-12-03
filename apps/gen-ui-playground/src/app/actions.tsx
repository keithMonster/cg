"use server";

import { createAI, getMutableAIState, streamUI } from "ai/rsc";
import { openai } from "@ai-sdk/openai";
import { ReactNode } from "react";
import { z } from "zod";
import { PizzaSlicer } from "@/components/pizza-slicer";

export interface ServerMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ClientMessage {
  id: string;
  role: "user" | "assistant";
  display: ReactNode;
}

export async function continueConversation(
  input: string
): Promise<ClientMessage> {
  "use server";

  const history = getMutableAIState();

  const result = await streamUI({
    model: openai("gpt-4o"),
    messages: [...history.get(), { role: "user", content: input }],
    text: ({ content, done }) => {
      if (done) {
        history.done((messages: ServerMessage[]) => [
          ...messages,
          { role: "assistant", content },
        ]);
      }
      return <div className="p-4 rounded-xl bg-card">{content}</div>;
    },
    tools: {
      slicePizza: {
        description:
          "Show an interactive pizza slicer when the user asks to cut, slice, or divide a pizza",
        parameters: z.object({
          slices: z
            .number()
            .describe("The number of slices to divide the pizza into"),
        }),
        generate: async function* ({ slices }) {
          yield <div className="text-muted-foreground">Creating pizza slicer...</div>;
          
          history.done((messages: ServerMessage[]) => [
            ...messages,
            {
              role: "assistant",
              content: `Created pizza slicer with ${slices} slices`,
            },
          ]);

          return <PizzaSlicer initialSlices={slices} />;
        },
      },
    },
  });

  return {
    id: Date.now().toString(),
    role: "assistant",
    display: result.value,
  };
}

export const AI = createAI<ServerMessage[], ClientMessage[]>({
  actions: {
    continueConversation,
  },
  initialAIState: [],
  initialUIState: [],
});
