"use client";

import { AI } from "./actions";

export function AIProvider({ children }: { children: React.ReactNode }) {
  return <AI>{children}</AI>;
}
