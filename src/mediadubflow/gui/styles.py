"""
Design system styling tokens and global QSS stylesheet for MediaDubFlow.

Strictly adheres to GUI and backend engineering rules:
- Zero drop-shadows or box-shadows.
- Flat, clean surfaces with crisp 1px borders.
- No excessive cards or containers.
- High contrast typography with native system fonts.
"""

from __future__ import annotations

# Color Palette Tokens
COLOR_CANVAS = "#f8fafc"  # Slate 50
COLOR_SURFACE = "#ffffff"  # Pure White
COLOR_BORDER = "#e2e8f0"  # Slate 200
COLOR_BORDER_INPUT = "#cbd5e1"  # Slate 300
COLOR_BORDER_FOCUS = "#2563eb"  # Blue 600
COLOR_ERROR = "#dc2626"  # Red 600
COLOR_TEXT_PRIMARY = "#0f172a"  # Slate 900
COLOR_TEXT_SECONDARY = "#64748b"  # Slate 500
COLOR_TEXT_MUTED = "#94a3b8"  # Slate 400
COLOR_PRIMARY = "#2563eb"  # Blue 600
COLOR_PRIMARY_HOVER = "#1d4ed8"  # Blue 700
COLOR_PRIMARY_PRESSED = "#1e40af"  # Blue 800
COLOR_SUCCESS = "#16a34a"  # Green 600

GLOBAL_QSS = f"""
/* Global Reset */
QMainWindow, QWidget {{
    background-color: {COLOR_CANVAS};
    color: {COLOR_TEXT_PRIMARY};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
}}

/* Header Bar */
QWidget#headerBar {{
    background-color: {COLOR_SURFACE};
    border-bottom: 1px solid {COLOR_BORDER};
    padding: 8px 16px;
}}

QLabel#brandLabel {{
    font-size: 16px;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
}}

QLabel#breadcrumbLabel {{
    font-size: 13px;
    color: {COLOR_TEXT_SECONDARY};
}}

/* Hero Section */
QLabel#heroTitle {{
    font-size: 24px;
    font-weight: 700;
    color: {COLOR_TEXT_PRIMARY};
}}

QLabel#heroSubtitle {{
    font-size: 14px;
    color: {COLOR_TEXT_SECONDARY};
}}

QLabel#sectionTitle {{
    font-size: 16px;
    font-weight: 600;
    color: {COLOR_TEXT_PRIMARY};
}}

/* Buttons */
QPushButton {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER_INPUT};
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: #f1f5f9;
    border-color: {COLOR_TEXT_MUTED};
}}

QPushButton:pressed {{
    background-color: #e2e8f0;
}}

QPushButton:disabled {{
    color: {COLOR_TEXT_MUTED};
    border-color: {COLOR_BORDER};
    background-color: #f8fafc;
}}

/* Primary Action Button */
QPushButton#primaryButton, QPushButton[primary="true"] {{
    background-color: {COLOR_PRIMARY};
    color: #ffffff;
    border: 1px solid {COLOR_PRIMARY_HOVER};
    font-weight: 600;
}}

QPushButton#primaryButton:hover, QPushButton[primary="true"]:hover {{
    background-color: {COLOR_PRIMARY_HOVER};
}}

QPushButton#primaryButton:pressed, QPushButton[primary="true"]:pressed {{
    background-color: {COLOR_PRIMARY_PRESSED};
}}

/* Text Inputs */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {COLOR_SURFACE};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER_INPUT};
    border-radius: 4px;
    padding: 6px 10px;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border: 1px solid {COLOR_BORDER_FOCUS};
}}

QLineEdit[hasError="true"] {{
    border: 1px solid {COLOR_ERROR};
}}

/* Tables */
QTableWidget {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    gridline-color: #f1f5f9;
    border-radius: 4px;
    outline: none;
}}

QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid #f1f5f9;
}}

QTableWidget::item:selected {{
    background-color: #eff6ff;
    color: {COLOR_TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {COLOR_CANVAS};
    color: {COLOR_TEXT_SECONDARY};
    font-weight: 600;
    padding: 8px;
    border: none;
    border-bottom: 1px solid {COLOR_BORDER};
}}

/* Status Bar */
QStatusBar {{
    background-color: {COLOR_SURFACE};
    border-top: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT_SECONDARY};
    font-size: 12px;
    padding: 4px 12px;
}}

/* Status Pills */
QLabel.statusBadge {{
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
}}

QLabel.statusCompleted {{
    border: 1px solid {COLOR_SUCCESS};
    color: {COLOR_SUCCESS};
    background-color: transparent;
}}

QLabel.statusProcessing {{
    border: 1px solid {COLOR_PRIMARY};
    color: {COLOR_PRIMARY};
    background-color: transparent;
}}

QLabel.statusFailed {{
    border: 1px solid {COLOR_ERROR};
    color: {COLOR_ERROR};
    background-color: transparent;
}}

QLabel.statusQueued {{
    border: 1px solid {COLOR_BORDER_INPUT};
    color: {COLOR_TEXT_SECONDARY};
    background-color: transparent;
}}

/* Progress Bar */
QProgressBar {{
    background-color: #e2e8f0;
    border: none;
    border-radius: 4px;
    text-align: center;
    font-size: 11px;
    font-weight: 600;
    color: {COLOR_TEXT_PRIMARY};
    min-height: 14px;
    max-height: 18px;
}}

QProgressBar::chunk {{
    background-color: {COLOR_PRIMARY};
    border-radius: 4px;
}}

/* Scroll Areas */
QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* Modern Cross-Platform ScrollBars (Mac and Windows compatible) */
QScrollBar:vertical {{
    background-color: transparent;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background-color: #cbd5e1;
    min-height: 24px;
    border-radius: 4px;
    margin: 1px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: #94a3b8;
}}

QScrollBar::handle:vertical:pressed {{
    background-color: #64748b;
}}

QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {{
    height: 0px;
    width: 0px;
    background: none;
    border: none;
}}

QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {{
    background: none;
    border: none;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background-color: transparent;
    height: 8px;
    margin: 0px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal {{
    background-color: #cbd5e1;
    min-width: 24px;
    border-radius: 4px;
    margin: 1px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: #94a3b8;
}}

QScrollBar::handle:horizontal:pressed {{
    background-color: #64748b;
}}

QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {{
    height: 0px;
    width: 0px;
    background: none;
    border: none;
}}

QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {{
    background: none;
    border: none;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

QScrollBar::corner {{
    background: transparent;
}}
"""
