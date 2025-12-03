"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ClientMessage, continueConversation } from "./actions";
import { useActions, useUIState } from "ai/rsc";

export default function Home() {
  const [input, setInput] = useState("");
  const [conversation, setConversation] = useUIState();
  const { continueConversation: continueConv } = useActions();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const value = input.trim();
    setInput("");

    setConversation((currentConversation: ClientMessage[]) => [
      ...currentConversation,
      {
        id: Date.now().toString(),
        role: "user",
        display: <div className="bg-muted p-4 rounded-xl text-right">{value}</div>,
      },
    ]);

    const message = await continueConv(value);
    setConversation((currentConversation: ClientMessage[]) => [
      ...currentConversation,
      message,
    ]);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-24 max-w-3xl mx-auto">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex mb-8">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Generative UI Playground&nbsp;
          <code className="font-mono font-bold">apps/gen-ui-playground</code>
        </p>
      </div>

      <div className="flex flex-col w-full flex-1 overflow-y-auto mb-8 space-y-4">
        {conversation.length === 0 && (
          <div className="text-center text-muted-foreground mt-20">
            <p className="mb-2">Welcome to Generative UI!</p>
            <p className="text-sm">Try: "I want to slice a pizza for 8 people"</p>
          </div>
        )}
        {conversation.map((message: ClientMessage) => (
          <div key={message.id}>{message.display}</div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="w-full flex gap-2 sticky bottom-8">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask for a UI component..."
          className="flex-1"
        />
        <Button type="submit">Send</Button>
      </form>
    </main>
  );
}
