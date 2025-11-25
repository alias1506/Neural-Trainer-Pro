import React from 'react'

export default function Header() {
    return (
        <header className="h-14 flex-shrink-0 flex items-center justify-between px-6 shadow-md z-10" style={{ backgroundColor: '#404258', color: '#FFFFFF' }}>
            <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center backdrop-blur-sm">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                    </svg>
                </div>
                <h1 className="text-lg font-bold tracking-wide">NEURAL TRAINER PRO</h1>
            </div>
            <div className="flex items-center gap-4">
                <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 border border-white/20 text-xs font-medium text-white backdrop-blur-sm">
                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                    System Ready
                </div>
            </div>
        </header>
    )
}
