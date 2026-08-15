import React from 'react';
import { motion } from 'framer-motion';

const GhostIconButton = ({
  icon,
  onClick,
  variant = 'secondary',
  size = 'medium',
  theme = 'dark',
  borderRadius = '50%',
  disabled = false,
  type = 'button',
  ...props
}) => {
  const isPrimary = variant === 'primary';
  const isGhost = variant === 'ghost';

  /* color settings for light/dark theme */
  const bgColor = 'var(--ghost-bg)';
  const borderColor = 'var(--ghost-border)';
  const textColor = 'var(--ghost-text)';

  /* gradient color for light/dark theme */
  const gradient = 'var(--ghost-gradient)';

  // Inner button styling based on variants
  let buttonBgColor = bgColor;
  let buttonTextColor = textColor;
  let borderBgColor = borderColor;

  if (isPrimary) {
    buttonBgColor = borderColor;
    buttonTextColor = bgColor;
    borderBgColor = bgColor;
  } else if (isGhost) {
    buttonBgColor = 'transparent';
    buttonTextColor = textColor;
    borderBgColor = 'transparent';
  }

  // Size calculations
  const buttonSize =
    size === 'extra_large' ? '80px' :
      size === 'large' ? '48px' :
        size === 'small' ? '36px' :
          '42px';

  const iconSize =
    size === 'extra_large' ? 40 :
      size === 'large' ? 24 :
        size === 'small' ? 16 :
          20;

  const innerBorderRadius = borderRadius.toString().endsWith('%')
    ? borderRadius
    : `calc(${borderRadius} - 3px)`;

  // React.cloneElement to inject correct size to Lucide icons if not already set
  const clonedIcon = icon && React.isValidElement(icon)
    ? React.cloneElement(icon, { size: icon.props.size || iconSize })
    : icon;

  // Outer container scale animations
  const buttonVariants = {
    initial: { scale: 1, y: 0 },
    hover: {
      scale: 1.05,
      y: -2,
      transition: { type: "spring", stiffness: 400, damping: 15 }
    },
    tap: {
      scale: 0.95,
      y: 1,
      transition: { type: "spring", stiffness: 500, damping: 10 }
    }
  };

  // Micro-animation: Float the icon up and down like a gentle ghost
  const iconContainerVariants = {
    initial: { y: 0, rotate: 0 },
    hover: {
      y: [0, -3, 2, -2, 0],
      rotate: [0, -4, 4, -2, 0],
      transition: {
        duration: 2.2,
        repeat: Infinity,
        repeatType: "loop",
        ease: "easeInOut"
      }
    }
  };

  return (
    <motion.button
      type={type}
      onClick={disabled ? undefined : onClick}
      initial="initial"
      animate="initial"
      whileHover={disabled ? undefined : "hover"}
      whileTap={disabled ? undefined : "tap"}
      variants={buttonVariants}
      disabled={disabled}
      style={{
        position: 'relative',
        display: 'inline-flex',
        justifyContent: 'center',
        alignItems: 'center',
        width: buttonSize,
        height: buttonSize,
        border: 'none',
        background: 'transparent',
        padding: '3px', // border width
        borderRadius: borderRadius,
        cursor: disabled ? 'not-allowed' : 'pointer',
        boxShadow: disabled
          ? 'none'
          : 'var(--ghost-shadow)',
        overflow: 'hidden',
        outline: 'none',
        opacity: disabled ? 0.45 : 1,
      }}
      {...props}
    >
      {/* 1. Solid / Semi-transparent Border Background */}
      {!isGhost && (
        <div style={{ position: 'absolute', inset: 0, background: borderBgColor, zIndex: 0, borderRadius }} />
      )}

      {/* 2. Gradient Border / Glow Background (Fades in on hover) */}
      {!disabled && (
        <motion.div
          variants={{
            initial: { opacity: 0 },
            hover: { opacity: 1 }
          }}
          transition={{ duration: 0.25 }}
          style={{
            position: 'absolute',
            inset: 0,
            background: gradient,
            zIndex: 0,
            borderRadius,
            opacity: 0,
          }}
        />
      )}

      {/* 3. Inner Container */}
      <div
        style={{
          position: 'relative',
          zIndex: 1,
          width: '100%',
          height: '100%',
          background: isGhost ? 'transparent' : buttonBgColor,
          borderRadius: innerBorderRadius,
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          color: buttonTextColor,
          transition: 'background-color 0.2s ease, border-color 0.2s ease',
        }}
      >
        {/* Ghost hover glassmorphism background for the ghost variant */}
        {isGhost && !disabled && (
          <motion.div
            variants={{
              initial: { opacity: 0 },
              hover: { opacity: 1 }
            }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'absolute',
              inset: 0,
              background: 'var(--ghost-hover-bg)',
              backdropFilter: 'blur(4px)',
              borderRadius: innerBorderRadius,
              zIndex: -1,
              opacity: 0,
            }}
          />
        )}

        <motion.div
          variants={iconContainerVariants}
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          {clonedIcon}
        </motion.div>
      </div>
    </motion.button>
  );
};

export default GhostIconButton;
