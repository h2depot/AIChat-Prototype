import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const GhostTooltip = ({
  children,
  content,
  position = 'top', // 'top' | 'bottom' | 'left' | 'right'
  theme = 'dark',
  delay = 0.2,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef(null);

  const clearShowTimer = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  const showTooltip = () => {
    if (!content) return;

    clearShowTimer();

    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
      timeoutRef.current = null;
    }, delay * 1000);
  };

  const hideTooltip = () => {
    clearShowTimer();
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      clearShowTimer();
    };
  }, []);

  const bgColor = 'var(--ghost-tooltip-bg)';
  const borderColor = 'var(--ghost-tooltip-border)';
  const textColor = 'var(--ghost-tooltip-text)';
  const glowColor = 'var(--ghost-tooltip-glow)';

  const posConfig = {
    top: {
      containerStyle: { bottom: '100%', left: '50%', marginBottom: '10px' },
      containerMotion: { x: '-50%', y: 0 },
      tailStyle: {
        bottom: '-5px',
        left: '50%',
        marginLeft: '-5px',
        borderBottom: `1px solid ${borderColor}`,
        borderRight: `1px solid ${borderColor}`,
      },
      initialAnimation: { opacity: 0, y: 10, x: '-50%', scale: 0.9 },
      exitAnimation: { opacity: 0, y: -5, x: '-50%', scale: 0.95 },
    },

    bottom: {
      containerStyle: { top: '100%', left: '50%', marginTop: '10px' },
      containerMotion: { x: '-50%', y: 0 },
      tailStyle: {
        top: '-5px',
        left: '50%',
        marginLeft: '-5px',
        borderTop: `1px solid ${borderColor}`,
        borderLeft: `1px solid ${borderColor}`,
      },
      initialAnimation: { opacity: 0, y: -10, x: '-50%', scale: 0.9 },
      exitAnimation: { opacity: 0, y: 5, x: '-50%', scale: 0.95 },
    },

    left: {
      containerStyle: { right: '100%', top: '50%', marginRight: '10px' },
      containerMotion: { x: 0, y: '-50%' },
      tailStyle: {
        right: '-5px',
        top: '50%',
        marginTop: '-5px',
        borderTop: `1px solid ${borderColor}`,
        borderRight: `1px solid ${borderColor}`,
      },
      initialAnimation: { opacity: 0, x: 10, y: '-50%', scale: 0.9 },
      exitAnimation: { opacity: 0, x: -5, y: '-50%', scale: 0.95 },
    },

    right: {
      containerStyle: { left: '100%', top: '50%', marginLeft: '10px' },
      containerMotion: { x: 0, y: '-50%' },
      tailStyle: {
        left: '-5px',
        top: '50%',
        marginTop: '-5px',
        borderBottom: `1px solid ${borderColor}`,
        borderLeft: `1px solid ${borderColor}`,
      },
      initialAnimation: { opacity: 0, x: -10, y: '-50%', scale: 0.9 },
      exitAnimation: { opacity: 0, x: 5, y: '-50%', scale: 0.95 },
    },
  };

  const config = posConfig[position] || posConfig.top;

  return (
    <div
      style={{
        position: 'relative',
        display: 'inline-flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
    >
      {children}

      <AnimatePresence>
        {isVisible && content && (
          <motion.div
            initial={config.initialAnimation}
            animate={{
              opacity: 1,
              x: config.containerMotion.x,
              y: config.containerMotion.y,
              scale: 1,
            }}
            exit={{
              ...config.exitAnimation,
              transition: { duration: 0.25, ease: 'easeIn' },
            }}
            transition={{
              type: 'spring',
              stiffness: 400,
              damping: 20,
            }}
            style={{
              position: 'absolute',
              zIndex: 1000,
              pointerEvents: 'none',
              ...config.containerStyle,
            }}
          >
            <motion.div
              animate={{
                y: [0, -3, 0],
              }}
              transition={{
                duration: 2.5,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
              style={{
                position: 'relative',
                background: bgColor,
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                border: `1px solid ${borderColor}`,
                borderRadius: '12px',
                padding: '8px 14px',
                color: textColor,
                fontSize: '12px',
                fontWeight: 600,
                whiteSpace: 'nowrap',
                boxShadow: `0 8px 20px rgba(0,0,0,0.15), 0 0 10px ${glowColor}`,
              }}
            >
              <span style={{ position: 'relative', zIndex: 1 }}>
                {content}
              </span>

              <div
                aria-hidden="true"
                style={{
                  position: 'absolute',
                  width: '10px',
                  height: '10px',
                  background: bgColor,
                  backdropFilter: 'blur(8px)',
                  WebkitBackdropFilter: 'blur(8px)',
                  transform: 'rotate(45deg)',
                  zIndex: 0,
                  ...config.tailStyle,
                }}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GhostTooltip;