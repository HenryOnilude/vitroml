/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{svelte,js,ts}'],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '"General Sans"',
          '-apple-system',
          'BlinkMacSystemFont',
          'Segoe UI',
          'sans-serif',
        ],
        mono: [
          '"JetBrains Mono"',
          'ui-monospace',
          'SFMono-Regular',
          'Menlo',
          'monospace',
        ],
        // Transitional serif for claims/headlines — the "document of record" voice
        // (Georgia is the closest system serif to Baskerville's authority profile)
        serif: ['Georgia', 'Charter', '"Times New Roman"', 'serif'],
      },
      colors: {
        canvas: {
          DEFAULT: '#f5f5f3',     // warm alabaster instrument base
          surface: '#ffffff',
          elevated: '#eeeeec',
          hover: '#e4e4e0',
        },
        hairline: {
          DEFAULT: '#dcdcd6',     // technical hairline
          subtle: '#eeeeec',
          bright: '#bfc0ba',
        },
        ink: {
          DEFAULT: '#111215',
          secondary: '#4b4e56',
          tertiary: '#5f6370',
          muted: '#767a85',
          faint: '#a6abb8',
        },
        accent: {
          DEFAULT: '#111215',
          bronze: '#b45309',
          bronzeLight: '#fef3c7',
          blue: '#2563eb',
          rose: '#e11d48',
          rust: '#b91c1c',        // ablation cut / severed pathway
          emerald: '#047857',
        },
        // Crameri vik diverging ramp: residual-stream activations
        vik: {
          neg100: '#001261',
          neg50: '#307da6',
          zero: '#ece5e0',
          pos50: '#c37243',
          pos100: '#590008',
        },
        // cividis sequential ramp: attention probabilities 0→1
        cividis: {
          0: '#00224e',
          25: '#434e6c',
          50: '#7d7c78',
          75: '#bcae6c',
          100: '#fee838',
        },
      },
      boxShadow: {
        subtle: '0 1px 2px 0 rgba(17, 18, 21, 0.04)',
        instrument: '0 1px 2px rgba(17,18,21,0.04), 0 0 0 1px #dcdcd6',
        inset: 'inset 0 1px 2px rgba(17, 18, 21, 0.06)',
        luxe: '0 1px 3px 0 rgba(17,18,21,0.05), 0 4px 12px -2px rgba(17,18,21,0.04)',
      },
      borderRadius: {
        sm: '3px',
        DEFAULT: '3px',           // industrial instrument corners
        md: '3px',
      },
    },
  },
  plugins: [],
};
