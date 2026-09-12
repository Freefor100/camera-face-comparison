"""桌面应用的深色视觉主题。"""

APP_STYLE_SHEET = """
QMainWindow, QWidget {
    background: #0b1020;
    color: #e8edf8;
    font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
    font-size: 14px;
}
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
#pageHint { color: #a9b7ff; background: #182345; border: 1px solid #31457d; border-radius: 14px; padding: 8px 12px; }
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
#controlCaption, #healthCaption { color: #7f8eae; font-size: 11px; font-weight: 700; }
#healthStrip { background: transparent; }
#healthCard {
    background: #121a2e;
    border: 1px solid #263452;
    border-radius: 10px;
    min-width: 150px;
}
#healthValue { color: #edf2ff; font-size: 14px; font-weight: 600; }
#healthValue[state="warning"] { color: #ffbd72; }
#recognitionWorkbench, #libraryWorkbench { background: transparent; border: 0; }
#previewCard, #peoplePanel, #personDetails {
    background: #11182a;
    border: 1px solid #263452;
    border-radius: 14px;
}
#cardTitle { color: #f0f4ff; font-size: 16px; font-weight: 700; }
#cardMeta { color: #7f8eae; }
#previewSurface {
    background: #050812;
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
    background: #070b17;
    border: 1px solid #263452;
    border-radius: 10px;
    color: #74809a;
}
#resultTitle, #detailTitle { color: #eef2ff; font-size: 18px; font-weight: 700; }
#statusText, #metricText { color: #a8b3c9; }
#qualityText { color: #ffcf8a; }
#integrityText { color: #77e1b5; }
#integrityText[state="warning"] { color: #ffbd72; }
#peopleList {
    background: #0d1425;
    border: 1px solid #263452;
    border-radius: 12px;
    padding: 6px;
}
#peopleList::item { padding: 12px; margin: 3px; border-radius: 8px; }
#peopleList::item:selected { background: #26375e; color: #ffffff; }
#sampleScroll { background: transparent; border: 0; }
#sampleContainer { background: transparent; }
#sampleCard { background: #18213a; border: 1px solid #2b3b61; border-radius: 10px; }
#sampleThumb { background: #080d1a; border-radius: 7px; color: #7785a1; }
#sampleCaption { color: #9daac1; font-size: 12px; }
#sectionDivider { color: #273654; }
"""
