"""桌面应用的深色视觉主题。"""

APP_STYLE_SHEET = """
QMainWindow, #recognitionPage, #libraryPage, QTabWidget {
    background: #0b1020;
}
QMainWindow QWidget {
    color: #e8edf8;
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}
QLabel { background: transparent; }
QTabWidget::pane { border: 0; }
QTabBar::tab {
    background: transparent;
    color: #8d99b3;
    padding: 14px 22px;
    margin: 0 4px;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected { color: #f4f7ff; border-bottom-color: #7c8cff; }
#pageTitle { color: #f8faff; font-size: 28px; font-weight: 700; }
#pageSubtitle { color: #98a4bd; font-size: 14px; padding-bottom: 4px; }
#headerStatus, #integrityText { color: #96a4c0; font-size: 12px; font-weight: 600; }
QPushButton {
    background: #202b48;
    border: 1px solid #344467;
    border-radius: 8px;
    color: #edf2ff;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #2a385c; border-color: #7084d9; }
QPushButton:pressed { background: #18213a; }
QPushButton:disabled { background: #151c30; border-color: #222c46; color: #65708b; }
QComboBox, QLineEdit {
    background: #131b30;
    border: 1px solid #344467;
    border-radius: 8px;
    padding: 8px 10px;
    min-height: 20px;
    color: #edf2ff;
}
#controlCaption { color: #7f8eae; font-size: 11px; font-weight: 700; }
#recognitionWorkbench, #libraryWorkbench { background: transparent; border: 0; }
#previewCard, #peoplePanel, #personDetails {
    background: #11182a;
    border: 1px solid #263452;
    border-radius: 14px;
}
#cardTitle { color: #f0f4ff; font-size: 16px; font-weight: 700; }
#cardMeta { color: #7f8eae; }
#previewSurface {
    background: #02040a;
    border: 1px solid #263452;
    border-radius: 12px;
    color: #74809a;
}
#resultCard {
    background: #121a2e;
    border: 1px solid #263452;
    border-radius: 14px;
}
#resultPreviewSurface {
    background: #02040a;
    border: 1px solid #263452;
    border-radius: 10px;
    color: #74809a;
}
#resultTitle, #detailTitle { color: #eef2ff; font-size: 18px; font-weight: 700; }
#resultName { color: #f8faff; font-size: 20px; font-weight: 650; }
#statusText, #metricText { color: #a8b3c9; }
#metricItem { background: transparent; }
#metricCaption { color: #74829f; font-size: 11px; font-weight: 600; }
#metricValue { color: #e7ecf8; font-size: 14px; font-weight: 600; }
#qualityText { color: #ffcf8a; }
#integrityText { color: #77e1b5; }
#integrityText[state="warning"] { color: #ffbd72; }
#peopleList {
    background: #0d1425;
    border: 1px solid #263452;
    border-radius: 12px;
    padding: 6px;
}
#peopleList::item { padding: 8px 10px; margin: 2px; border-radius: 7px; }
#peopleList::item:selected { background: #26375e; color: #ffffff; }
#sampleScroll, #sampleScroll QWidget { background: transparent; border: 0; }
#sampleContainer { background: transparent; }
#sampleCard { background: #18213a; border: 1px solid #2b3b61; border-radius: 10px; }
#sampleThumb { background: #050812; border-radius: 7px; color: #7785a1; }
#sampleCaption { color: #9daac1; font-size: 12px; }
#sectionDivider { color: #273654; }
QScrollBar:vertical {
    background: #0d1425;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #354566;
    border-radius: 5px;
    min-height: 28px;
}
QScrollBar::handle:vertical:hover { background: #4a5f8a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""
