import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Ghost } from 'lucide-react';
import GhostTooltip from './GhostTooltip';

const GhostTextField = ({
  value,
  onChange,
  placeholder = "なにか入力して...",
  theme = 'dark',
  disabled = false,
  icon, // 左側に置くアイコン (例: <Search size={18} /> など)
  showClearButton = true,
  borderRadius = '16px',
  width = '100%',
  ...props
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const inputRef = useRef(null);

  const bgColor = 'var(--ghost-bg)';
  const borderColor = 'var(--ghost-border)';
  const textColor = 'var(--ghost-text)';

  const gradient = 'var(--ghost-gradient)';

  const handleClear = () => {
    if (onChange) {
      onChange({ target: { value: '' } });
    }
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  return (
    <div style={{ width: width, position: 'relative', boxSizing: 'border-box' }}>
      <motion.div
        animate={{
          y: isFocused ? -2 : 0,
          scale: isFocused ? 1.01 : 1,
          boxShadow: isFocused
            ? '0 8px 24px var(--ghost-glow)'
            : isHovered
              ? 'var(--ghost-shadow-input-focus)'
              : 'var(--ghost-shadow-input)',
        }}
        transition={{ type: "spring", stiffness: 400, damping: 25 }}
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
        style={{
          position: 'relative',
          borderRadius: borderRadius,
          padding: '4px', // 外枠の太さ（グラデーション境界線）
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          transition: 'box-shadow 0.3s ease',
        }}
      >
        {/* Solid Border Background */}
        <div style={{ position: 'absolute', inset: 0, background: borderColor, zIndex: 0 }} />

        {/* Gradient Border Background (Fades in on hover / full on focus) */}
        <motion.div
          animate={{
            opacity: isFocused ? 1 : isHovered ? 0.5 : 0
          }}
          transition={{ duration: 0.25 }}
          style={{ position: 'absolute', inset: 0, background: gradient, zIndex: 0 }}
        />

        {/* Inner Container */}
        <div
          style={{
            position: 'relative',
            zIndex: 1,
            background: bgColor,
            borderRadius: `calc(${borderRadius} - 4px)`,
            padding: '10px 16px',
            display: 'flex',
            alignItems: 'center',
            width: '100%',
            gap: '10px',
            transition: 'background-color 0.3s ease',
          }}
        >
          {/* 左側のアイコン */}
          {icon && (
            <div style={{ display: 'flex', alignItems: 'center', color: textColor, opacity: 0.7 }}>
              {icon}
            </div>
          )}

          {/* インプット要素 */}
          <input
            ref={inputRef}
            type="text"
            value={value}
            onChange={onChange}
            placeholder={placeholder}
            disabled={disabled}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            style={{
              flex: 1,
              minWidth: 0,
              border: 'none',
              background: 'transparent',
              outline: 'none',
              color: textColor,
              fontWeight: 700,
              fontSize: '15px',
              fontFamily: 'inherit',
              padding: 0,
              margin: 0,
              width: '100%',
            }}
            className="ghost-text-field-input"
            {...props}
          />

          {/* クリアおばけボタン */}
          <AnimatePresence>
            {showClearButton && value && !disabled && (
              <GhostTooltip content="入力をクリア" position="top">
                <motion.button
                  type="button"
                  onClick={handleClear}
                  initial={{ opacity: 0, scale: 0.5, rotate: -45 }}
                  animate={{ opacity: 1, scale: 1, rotate: 0 }}
                  exit={{ opacity: 0, scale: 0.5, rotate: 45 }}
                  whileHover={{ scale: 1.15, rotate: [0, -10, 10, 0] }}
                  transition={{
                    type: "spring",
                    stiffness: 500,
                    damping: 15,
                    rotate: { duration: 0.5 }
                  }}
                  style={{
                    border: 'none',
                    background: 'transparent',
                    padding: 0,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flex: '0 0 16px',
                    width: '16px',
                    height: '16px',
                    color: textColor,
                    outline: 'none',
                  }}
                  aria-label="入力をクリア"
                >
                  {/* フワフワと浮く可愛いおばけアイコン */}
                  <motion.div
                    animate={{
                      y: [0, -3, 0],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 1.5,
                      ease: "easeInOut"
                    }}
                  >
                    <Ghost size={12} strokeWidth={2.25} />
                  </motion.div>
                </motion.button>
              </GhostTooltip>
            )}
          </AnimatePresence>
        </div>
      </motion.div>

      {/* プレースホルダーのカスタムスタイルと disabled スタイル */}
      <style>{`
        .ghost-text-field-input::placeholder {
          color: var(--ghost-placeholder) !important;
          opacity: 1;
        }
        .ghost-text-field-input:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default GhostTextField;
