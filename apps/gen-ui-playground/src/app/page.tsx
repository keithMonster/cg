"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { streamComponent } from "./actions";
import { PizzaSlicer } from "@/components/pizza-slicer";

export default function Home() {
  const [messages, setMessages] = useState<React.ReactNode[]>([]);
  const [input, setInput] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = <div key={messages.length} className="bg-muted p-4 rounded-xl mb-4 text-right self-end">{input}</div>;
    setMessages((prev) => [...prev, userMessage]);
    
    const currentInput = input.trim().toLowerCase();
    setInput("");

    try {
      // Simple pattern matching for pizza slicing
      if (currentInput.includes("pizza") || currentInput.includes("slice")) {
        const match = currentInput.match(/(\d+)/);
        const slices = match ? parseInt(match[1]) : 4;
        
        setMessages((prev) => [...prev, 
          <div key={prev.length} className="mb-4">
            <PizzaSlicer initialSlices={slices} />
          </div>
        ]);
      } else {
        // For other queries, use AI
        const textContent = await streamComponent(currentInput);
        
        setMessages((prev) => [...prev, 
          <div key={prev.length} className="bg-card p-4 rounded-xl mb-4">{textContent}</div>
        ]);
      }
    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, <div key={prev.length} className="text-red-500">Error: {String(error)}</div>]);
    }
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
        {messages.length === 0 && (
           <div className="text-center text-muted-foreground mt-20">
             <p>Try asking: "I want to slice a pizza for 3 people"</p>
           </div>
        )}
        {messages}
      </div>

      <form onSubmit={handleSubmit} className="w-full flex gap-2 sticky bottom-8">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask for a UI..."
          className="flex-1"
        />
        <Button type="submit">Send</Button>
      </form>
    </main>
  );
}
