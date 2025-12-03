"use server";

import { streamText } from "ai";
import { openai } from "@ai-sdk/openai";



export async function streamComponent(prompt: string) {
  const { textStream } = await streamText({
    model: openai("gpt-4o"),
    prompt: prompt,
  });

  let fullText = "";
  for await (const delta of textStream) {
    fullText += delta;
  }

  return fullText;
}

