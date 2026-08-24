import { invoke } from "@tauri-apps/api/core";
import { create } from "zustand";

export const useLlmStore = create((set, get) => ({
  initialized: false,
  initializing: false,
  generating: false,
  error: null,
  messages: [],

  initialize: async () => {
    if (get().initialized || get().initializing) return;

    set({ initializing: true, error: null });
    try {
      await invoke("initialize");
      set({ initialized: true });
    } catch (error) {
      set({ error: String(error) });
    } finally {
      set({ initializing: false });
    }
  },

  generate: async (prompt) => {
    const text = prompt.trim();
    if (!text || get().generating) return;
    if (!get().initialized) {
      set({ error: "LLM is not initialized" });
      return;
    }

    set((state) => ({
      generating: true,
      error: null,
      messages: [...state.messages, { role: "user", content: text }],
    }));

    try {
      const response = await invoke("generate", { prompt: text });
      set((state) => ({
        messages: [
          ...state.messages,
          { role: "assistant", content: response },
        ],
      }));
      return response;
    } catch (error) {
      set({ error: String(error) });
    } finally {
      set({ generating: false });
    }
  },

  clearMessages: async () => {
    await invoke("clear_chat");
    set({ messages: [], error: null });
  },
  clearError: () => set({ error: null }),
}));
