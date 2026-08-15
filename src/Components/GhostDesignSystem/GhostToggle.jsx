import React from 'react';
import { motion } from 'framer-motion';

const GhostToggle = ({ isOn, onToggle, label, theme = 'dark', scale = 1, disabled = false }) => {
  const handleToggle = (e) => {
    if (disabled) {
      e.stopPropagation();
      return;
    }
    if (onToggle) {
      onToggle();
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: `${12 * scale}px`,
        cursor: disabled ? 'not-allowed' : 'pointer',
        pointerEvents: disabled ? 'none' : 'auto'
      }}
      onClick={handleToggle}
    >
      {label && <span style={{ fontSize: `${14 * scale}px`, opacity: 0.8 }}>{label}</span>}
      <div
        style={{
          width: `${72 * scale}px`,
          height: `${36 * scale}px`,
          background: isOn
            ? 'var(--ghost-gradient)'
            : 'var(--ghost-bg)',
          borderRadius: `${10 * scale}px`,
          border: `${6 * scale}px solid var(--ghost-border)`,
          padding: `${4 * scale}px`,
          display: 'flex',
          justifyContent: isOn ? 'flex-end' : 'flex-start',
          alignItems: 'center',
          boxShadow: isOn
            ? '0 4px 12px var(--ghost-glow)'
            : 'var(--ghost-toggle-shadow-off)',
          transition: 'all 0.3s ease',
          opacity: disabled ? 0.5 : 1
        }}
      >
        <motion.div
          layout
          whileHover={disabled ? undefined : { scale: 1.1 }}
          whileTap={disabled ? undefined : { scale: 0.95 }}
          transition={{
            type: "spring",
            stiffness: 400,
            damping: 30
          }}
          style={{
            width: `${30 * scale}px`,
            height: `${30 * scale}px`,
            background: 'var(--ghost-border)',
            borderRadius: `${6 * scale}px`,
            boxShadow: 'var(--ghost-toggle-knob-shadow)'
          }}
        />
      </div>
    </div >
  );
};

export default GhostToggle;
