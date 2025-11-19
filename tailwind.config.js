/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5f9ff',
          100: '#eaf1ff',
          200: '#d6e4ff',
          300: '#adc8ff',
          400: '#84a9ff',
          500: '#3f8cff',
          600: '#2f6bd9',
          700: '#234fb0',
          800: '#1a3a8a',
          900: '#132a66'
        }
      }
    }
  },
  plugins: []
}