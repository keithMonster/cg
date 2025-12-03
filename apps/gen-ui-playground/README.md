# Generative UI Playground

This is a Next.js application demonstrating **Generative UI** (Server-Driven UI with Vercel AI SDK).

## Setup

1.  **Install Dependencies**:
    ```bash
    npm install
    ```

2.  **Configure Environment**:
    Create a `.env.local` file in this directory:
    ```env
    OPENAI_API_KEY=sk-proj-your-key-here
    ```

3.  **Run Development Server**:
    ```bash
    npm run dev
    ```

## Demos

### 1. The Pizza Slicer
*   **Prompt**: "I want to slice a pizza for 3 people"
*   **Result**: An interactive SVG component where you can adjust slices.
*   **Mechanism**: `streamUI` calls the `slicePizza` tool, which returns the `<PizzaSlicer />` client component.
