import React, { useState, useRef, useEffect } from "react";
import { GhostIconButton, GhostDropdown, GhostTextField, GhostToggle, GhostTooltip } from "../GhostDesignSystem";
import { useLlmStore } from "../store/llm";
import { Send, Bot, User, Sparkles, Sun, Moon, ArrowLeft } from "lucide-react";
import "./tabview.css";

export default function TabView() {
    const [selectedOption, setSelectedOption] = useState("rawtext1");
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

    const dropdownOptions = [
        { label: "rawtext1", value: "rawtext1" },
        { label: "rawtext2", value: "rawtext2" },
        { label: "rawtext3", value: "rawtext3" },
    ];

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

    const handleReset = () => {
        setMessages([]);
        setInputText("");
    };

    return (
        <div className="tab-view-container" data-theme={isDarkMode ? "dark" : "light"}>
            {/* Top Header */}
            <header className="tab-view-header">
                <div className="tab-view-header-title">
                    <span>Ghost Librarian</span>
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
                        onChange={(val) => setSelectedOption(val)}
                        placeholder="Select Mode"
                    />
                </div>
            </header>

            {/* Main Content Area */}
            <main className="tab-view-main">
                {messages.length === 0 ? (
                    <div className="tab-view-hero">
                        <h1 className="tab-view-hero-title">このPCに住むGhostと会話する</h1>
                        <p className="tab-view-hero-subtitle">
                            {initializing
                                ? "言語モデルをイニシャライズ中...👻"
                                : "このPCは物語をより楽しむためのお手伝いをするGhostが住んでいます。物語に関する質問をしてみて！"}
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
                                placeholder="この物語について聞きたいことはあるかな？"
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
                        {selectedOption}について話そう！ | AIモデルからの回答は間違っている可能性があります…ゴメンネ💦
                    </span>
                </section>
            </main>
        </div>
    );
}
