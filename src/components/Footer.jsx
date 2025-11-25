import React from 'react'

export default function Footer() {
    return (
        <footer className="h-10 flex-shrink-0 border-t border-gray-200 bg-white flex items-center justify-between px-6 text-xs text-gray-500">
            <div className="flex items-center gap-4">
                <span>&copy; 2025 Neural Trainer Pro</span>
                <span>v1.2.0</span>
            </div>
            <div className="flex items-center gap-4">
                <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-green-500"></span>
                    Backend Connected
                </span>
            </div>
        </footer>
    )
}
