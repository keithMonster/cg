"use server";

import { streamUI } from "ai/rsc";
import { openai } from "@ai-sdk/openai";
import { z } from "zod";
import { PizzaSlicer } from "@/components/pizza-slicer";

export async function streamComponent(prompt: string) {
  const result = await streamUI({
    model: openai("gpt-4o"),
    prompt: prompt,
    text: ({ content }) => <div>{content}</div>,
    tools: {
      slicePizza: {
        description: "Show a pizza slicer tool when the user asks to cut or slice a pizza.",
        parameters: z.object({
          initialSlices: z.number().default(1).describe("The initial number of slices to show"),
        }),
        generate: async ({ initialSlices }) => {
          return <PizzaSlicer initialSlices={initialSlices} />;
        },
      },
    },
  });

  return result.value;
}
