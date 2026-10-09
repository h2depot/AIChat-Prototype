import React, { useState, useRef, useEffect } from "react";
import { GhostIconButton, GhostDropdown, GhostTextField, GhostToggle, GhostTooltip } from "../GhostDesignSystem";
import { useLlmStore } from "../store/llm";
import { Send, Bot, User, Sparkles, Sun, Moon, ArrowLeft } from "lucide-react";
import "./tabview.css";

export default function TabView() {
    const [selectedOption, setSelectedOption] = useState("none");
    const [inputText, setInputText] = useState("");
    const [messages, setMessages] = useState([]);
    const [isDarkMode, setIsDarkMode] = useState(true);
    const messagesEndRef = useRef(null);
    const initialize = useLlmStore((state) => state.initialize);
    const initializing = useLlmStore((state) => state.initializing);
    const initialized = useLlmStore((state) => state.initialized);
    const generate = useLlmStore((state) => state.generate);
    const generating = useLlmStore((state) => state.generating);
    const error = useLlmStore((state) => state.error);
    const clearError = useLlmStore((state) => state.clearError);
    const clearMessages = useLlmStore((state) => state.clearMessages);

    const dropdownOptions = [
        { label: "テンプレートなし", value: "none", prompt: "" },
        {
            label: "わかりやすく解説",
            value: "explain",
            prompt: "機械学習を身近な例で簡単に説明して。",
        },
        {
            label: "アイデアを出す",
            value: "ideas",
            prompt: "紙とペンで一人で遊べるアイデアを5つ教えて。",
        },
        {
            label: "文章を整える",
            value: "rewrite",
            prompt: "「もう少し説明してほしいです」を丁寧に言い換えて。",
        },
        { label: "違いを比べる", value: "compare", prompt: "メモと日記の違いを簡単に教えて。" },
        { label: "英語に翻訳", value: "translate", prompt: "「手伝ってくれてありがとう」を自然な英語にして。" },
    ];

    const handleTemplateSelect = (value) => {
        const template = dropdownOptions.find((option) => option.value === value);
        if (!template) return;
        setSelectedOption(value);
        setInputText(template.prompt);
    };

    useEffect(() => {
        document.documentElement.setAttribute("data-theme", isDarkMode ? "dark" : "light");
    }, [isDarkMode]);

    useEffect(() => {
        initialize();
    }, [initialize]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSend = async () => {
        const trimmed = inputText.trim();
        if (!trimmed || !initialized || generating) return;

        clearError();
        const userMsg = { id: Date.now(), sender: "user", text: trimmed };
        setMessages((prev) => [...prev, userMsg]);
        setInputText("");

        const response = await generate(trimmed);
        if (response) {
            const aiMsg = {
                id: Date.now() + 1,
                sender: "ai",
                text: response,
            };
            setMessages((prev) => [...prev, aiMsg]);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleReset = async () => {
        await clearMessages();
        setMessages([]);
        setInputText("");
    };

    return (
        <div className="tab-view-container" data-theme={isDarkMode ? "dark" : "light"}>
            {/* Top Header */}
            <header className="tab-view-header">
                <div className="tab-view-header-title">
                    <span>Local Model Chat</span>
                </div>
                <div className="tab-view-header-right">
                    <GhostTooltip content={isDarkMode ? "夜モード" : "昼モード"} position="bottom">
                        <div
                            style={{
                                display: "flex",
                                alignItems: "center",
                                gap: "8px",
                                marginRight: "8px",
                                color: "var(--ghost-text)",
                                cursor: "pointer",
                            }}
                        >
                            {isDarkMode ? <Moon size={18} /> : <Sun size={18} />}
                            <GhostToggle
                                isOn={isDarkMode}
                                onToggle={() => setIsDarkMode(!isDarkMode)}
                                scale={0.75}
                            />
                        </div>
                    </GhostTooltip>
                    <GhostDropdown
                        options={dropdownOptions}
                        value={selectedOption}
                        onChange={handleTemplateSelect}
                        placeholder="質問テンプレート"
                    />
                </div>
            </header>

            {/* Main Content Area */}
            <main className="tab-view-main">
                {messages.length === 0 ? (
                    <div className="tab-view-hero">
                        <h1 className="tab-view-hero-title">このPCのローカルモデルと会話する</h1>
                        <p className="tab-view-hero-subtitle">
                            {initializing
                                ? "言語モデルをイニシャライズ中...👻"
                                : "このPCのローカルモデルに質問してみて！"}
                        </p>
                    </div>
                ) : (
                    <>
                        <div className="tab-view-back-bar">
                            <GhostTooltip content="最初の画面に戻る" position="right">
                                <GhostIconButton
                                    icon={<ArrowLeft size={18} />}
                                    onClick={handleReset}
                                    variant="secondary"
                                    size="medium"
                                />
                            </GhostTooltip>
                        </div>
                        <div className="tab-view-messages">
                            {messages.map((msg) => (
                                <div key={msg.id} className={`chat-message ${msg.sender}`}>
                                    {msg.sender === "user" && (
                                        <div className="chat-avatar">
                                            <User size={18} />
                                        </div>
                                    )}
                                    <div className="chat-bubble">{msg.text}</div>
                                </div>
                            ))}
                            <div ref={messagesEndRef} />
                        </div>
                    </>
                )}

                {/* Input Bar Area - Centered at Bottom with GhostTextField & GhostIconButton */}
                <section className="tab-view-input-section">
                    {error && (
                        <div className="tab-view-error" role="alert">
                            <span>{error}</span>
                            <button type="button" onClick={clearError} aria-label="エラーを閉じる">
                                ×
                            </button>
                        </div>
                    )}
                    <div className="tab-view-input-bar">
                        <div style={{ flex: 1 }}>
                            <GhostTextField
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="聞きたいことを入力してね"
                            />
                        </div>
                        <GhostTooltip content="送信" position="top">
                            <GhostIconButton
                                icon={<Send size={18} />}
                                onClick={handleSend}
                                variant="primary"
                                size="medium"
                                disabled={!inputText.trim() || !initialized || generating}
                            />
                        </GhostTooltip>
                    </div>
                    <span className="tab-view-disclaimer">
                        AIモデルからの回答は間違っている可能性があります…ゴメンネ💦
                    </span>
                </section>
            </main>
        </div>
    );
}
