package com.tabbyml.tabby4eclipse.inlineCompletion;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.eclipse.jface.text.IPaintPositionManager;
import org.eclipse.jface.text.IPainter;
import org.eclipse.jface.text.ITextViewer;
import org.eclipse.jface.text.ITextViewerExtension2;
import org.eclipse.jface.text.Position;
import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.StyleRange;
import org.eclipse.swt.custom.StyledText;
import org.eclipse.swt.events.PaintEvent;
import org.eclipse.swt.events.PaintListener;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.FontData;
import org.eclipse.swt.graphics.FontMetrics;
import org.eclipse.swt.graphics.GC;
import org.eclipse.swt.graphics.GlyphMetrics;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.widgets.Display;

import com.tabbyml.tabby4eclipse.Logger;

public class InlineCompletionRenderer {
	private Logger logger = new Logger("InlineCompletionRenderer");
	private Map<ITextViewer, InlineCompletionItemPainter> painters = new HashMap<>();
	private ITextViewer currentTextViewer = null;
	private InlineCompletionItem currentCompletionItem = null;
	private String currentViewId = null;
	private Long currentDisplayAt = null;

	public void show(ITextViewer viewer, InlineCompletionItem item) {
		if (currentTextViewer != null) {
			getPainter(currentTextViewer).update(null);
		}
		currentTextViewer = viewer;
		currentCompletionItem = item;
		getPainter(viewer).update(item);
		
		currentDisplayAt = System.currentTimeMillis();
		
		String completionId = "no-cmpl-id";
		if (item.getEventId() != null && item.getEventId().getCompletionId()!= null) {
			completionId = item.getEventId().getCompletionId().replace("cmpl-", "");
		}
		currentViewId = String.format("view-%s-at-%d", completionId, currentDisplayAt);
	}

	public void hide() {
		if (currentTextViewer != null) {
			getPainter(currentTextViewer).update(null);
			currentTextViewer = null;
		}
		currentCompletionItem = null;
		currentViewId = null;
		currentDisplayAt = null;
	}

	public ITextViewer getCurrentTextViewer() {
		return currentTextViewer;
	}

	public InlineCompletionItem getCurrentCompletionItem() {
		return currentCompletionItem;
	}
	
	public String getCurrentViewId() {
		return currentViewId;
	}
	
	public Long getCurrentDisplayedTime() {
		if (currentDisplayAt != null) {
			return System.currentTimeMillis() - currentDisplayAt;
		}
		return null;
	}

	private InlineCompletionItemPainter getPainter(ITextViewer viewer) {
		InlineCompletionItemPainter painter = painters.get(viewer);
		if (painter == null) {
			painter = new InlineCompletionItemPainter(viewer);
			painters.put(viewer, painter);
		}
		return painter;
	}

	private class InlineCompletionItemPainter implements IPainter, PaintListener {
		private ITextViewer viewer;

		private InlineCompletionItem item;
		private int offset;

		private IPaintPositionManager positionManager;
		private Font font;
		private List<ModifiedLineVerticalIndent> modifiedLinesVerticalIndent = new ArrayList<>();
		private List<GlyphMetrics> modifiedGlyphMetrics = new ArrayList<>();
		private List<Consumer<GC>> paintFunctions = new ArrayList<>();

		public InlineCompletionItemPainter(ITextViewer viewer) {
			this.viewer = viewer;
			getDisplay().syncExec(() -> {
				((ITextViewerExtension2) this.viewer).addPainter(this);
				getWidget().addPaintListener(this);
			});
		}

		public void update(InlineCompletionItem item) {
			if (this.item != item) {
				this.item = item;
				getDisplay().syncExec(() -> {
					this.offset = getWidget().getCaretOffset();
					cleanup();
					setupPainting();
					getWidget().redraw();
				});
			}
		}

		@Override
		public void paintControl(PaintEvent event) {
			paintFunctions.forEach((fn) -> {
				fn.accept(event.gc);
			});
		}

		@Override
		public void paint(int reason) {
		}

		@Override
		public void deactivate(boolean redraw) {
		}

		@Override
		public void dispose() {
			logger.debug("Painter dispose called.");
			getWidget().removePaintListener(this);
			if (font != null) {
				font.dispose();
				font = null;
			}
		}

		@Override
		public void setPositionManager(IPaintPositionManager manager) {
			this.positionManager = manager;
		}

		private StyledText getWidget() {
			return viewer.getTextWidget();
		}

		private Display getDisplay() {
			return getWidget().getDisplay();
		}

		private void cleanup() {
			try {
				paintFunctions.clear();

				StyledText widget = getWidget();
				modifiedLinesVerticalIndent.forEach((modifiedLineVerticalIndent) -> {
					Position position = modifiedLineVerticalIndent.position;
					int line = widget.getLineAtOffset(position.getOffset());
					positionManager.unmanagePosition(position);
					int indent = modifiedLineVerticalIndent.indent;
					int modifiedIndent = modifiedLineVerticalIndent.modifiedIndent;
					// Find the line to restore the indent
					int lineToRestore = -1;
					int delta = 0;
					while (delta < widget.getLineCount()) {
						lineToRestore = line + delta;
						if (lineToRestore >= 0 && lineToRestore < widget.getLineCount()
								&& widget.getLineVerticalIndent(lineToRestore) == modifiedIndent) {
							break;
						}
						lineToRestore = line - delta;
						if (lineToRestore >= 0 && lineToRestore < widget.getLineCount()
								&& widget.getLineVerticalIndent(lineToRestore) == modifiedIndent) {
							break;
						}
						delta++;
					}
					if (lineToRestore >= 0 && lineToRestore < widget.getLineCount()) {
						widget.setLineVerticalIndent(lineToRestore, indent);
						logger.debug("Restore LineVerticalIndent: " + lineToRestore + " -> " + indent);
					}
				});
				modifiedLinesVerticalIndent.clear();

				StyleRange[] styleRanges = getWidget().getStyleRanges();
				for (StyleRange styleRange : styleRanges) {
					if (modifiedGlyphMetrics.contains(styleRange.metrics)) {
						styleRange.metrics = null;
						getWidget().setStyleRange(styleRange);
						logger.debug("Restore StyleRange:" + styleRange.start + " -> " + styleRange.metrics);
					}
				}
				modifiedGlyphMetrics.clear();
			} catch (Exception e) {
				logger.error("Failed to cleanup renderer.", e);
			}
		}

		private void setupPainting() {
			if (item == null) {
				return;
			}
			StyledText widget = getWidget();

			int prefixReplaceLength = item.getReplaceRange().getPrefixLength();
			int suffixReplaceLength = item.getReplaceRange().getSuffixLength();
			String text = item.getInsertText().substring(prefixReplaceLength);
			if (text.isEmpty()) {
				return;
			}
			logger.debug("Begin setupPainting...");

			int currentLineEndOffset;
			int nextLineNumber = widget.getLineAtOffset(offset) + 1;
			if (nextLineNumber < widget.getLineCount()) {
				currentLineEndOffset = widget.getOffsetAtLine(nextLineNumber) - 1;
			} else {
				currentLineEndOffset = widget.getCharCount() - 1;
			}
			String currentLineSuffix = "";
			if (offset < widget.getCharCount() && offset < currentLineEndOffset) {
				currentLineSuffix = widget.getText(offset, currentLineEndOffset);
			}

			String textCurrentLine;
			String textSuffixLines;
			int firstLineBreakIndex = text.indexOf("\n");
			if (firstLineBreakIndex == -1) {
				textCurrentLine = text;
				textSuffixLines = "";
			} else {
				textCurrentLine = text.substring(0, firstLineBreakIndex);
				textSuffixLines = text.substring(firstLineBreakIndex + 1);
			}

			if (suffixReplaceLength == 0 || currentLineSuffix.isEmpty()) {
				// No replace range to handle
				if (textSuffixLines.isEmpty()) {
					drawInsertPartText(offset, textCurrentLine);
				} else {
					if (!textCurrentLine.isEmpty()) {
						drawOverwriteText(offset, textCurrentLine);
					}
					drawSuffixLines(offset, textSuffixLines + currentLineSuffix);
				}
			} else if (suffixReplaceLength == 1) {
				// Replace range contains one char
				char replaceChar = currentLineSuffix.charAt(0);
				int replaceCharIndex = textCurrentLine.indexOf(replaceChar);
				if (replaceCharIndex > 0) {
					// If textCurrentLine contain the replaceChar
					// InsertPart is substring of textCurrentLine that before the replaceChar
					// AppendPart is substring of textCurrentLine that after the replaceChar
					String insertPart = textCurrentLine.substring(0, replaceCharIndex);
					String appendPart = textCurrentLine.substring(replaceCharIndex + 1);
					if (!insertPart.isEmpty()) {
						drawInsertPartText(offset, insertPart);
					}
					if (!appendPart.isEmpty()) {
						if (textSuffixLines.isEmpty()) {
							drawInsertPartText(offset + 1, appendPart);
						} else {
							drawOverwriteText(offset + 1, appendPart);
						}
					}
				} else {
					drawReplacePartText(offset, textCurrentLine, currentLineSuffix.substring(0, 1));
				}
				if (!textSuffixLines.isEmpty()) {
					drawSuffixLines(offset, textSuffixLines + currentLineSuffix.substring(1));
				}
			} else {
				// Replace range contains multiple chars
				if (textSuffixLines.isEmpty()) {
					drawReplacePartText(offset, textCurrentLine, currentLineSuffix.substring(0, suffixReplaceLength));
				} else {
					if (!textCurrentLine.isEmpty()) {
						drawOverwriteText(offset, textCurrentLine);
					}
					drawSuffixLines(offset, textSuffixLines + currentLineSuffix.substring(suffixReplaceLength));
				}
			}
			logger.debug("End setupPainting.");
		}

		private void drawOverwriteText(int offset, String text) {
			logger.debug("drawCurrentLineText:" + offset + ":" + text);
			StyledText widget = getWidget();
			TextWithTabs textWithTabs = splitLeadingTabs(text);

			paintFunctions.add((gc) -> {
				// Draw ghost text
				setStyleToGhostText(gc);
				int spaceWidth = gc.textExtent(" ").x;
				int tabWidth = textWithTabs.tabs * widget.getTabs() * spaceWidth;
				Point location = widget.getLocationAtOffset(offset);
				gc.drawString(textWithTabs.text, location.x + tabWidth, location.y);
			});
		}

		private void drawInsertPartText(int offset, String text) {
			drawReplacePartText(offset, text, "");
		}

		private void drawReplacePartText(int offset, String text, String replacedText) {
			logger.debug("drawReplacePartText:" + offset + ":" + text + ":" + replacedText);
			StyledText widget = getWidget();
			TextWithTabs textWithTabs = splitLeadingTabs(text);

			int targetOffset = offset + replacedText.length();
			if (targetOffset >= widget.getCharCount()) {
				// End of document, draw the ghost text only
				paintFunctions.add((gc) -> {
					// Draw ghost text
					setStyleToGhostText(gc);
					int spaceWidth = gc.textExtent(" ").x;
					int tabWidth = textWithTabs.tabs * widget.getTabs() * spaceWidth;
					Point location = widget.getLocationAtOffset(offset);
					gc.drawString(textWithTabs.text, location.x + tabWidth, location.y);
				});
				
			} else {
				// otherwise, draw the ghost text, and move target char after the ghost text
				String targetChar = widget.getText(targetOffset, targetOffset);
				StyleRange originStyleRange;
				if (widget.getStyleRangeAtOffset(targetOffset) != null) {
					originStyleRange = widget.getStyleRangeAtOffset(targetOffset);
					logger.debug("Find origin StyleRange:" + originStyleRange.start + " -> " + originStyleRange.metrics);
				} else {
					originStyleRange = new StyleRange();
					originStyleRange.start = targetOffset;
					originStyleRange.length = 1;
					logger.debug("Create StyleRange:" + originStyleRange.start + " -> " + originStyleRange.metrics);
				}

				paintFunctions.add((gc) -> {
					// Draw ghost text
					setStyleToGhostText(gc);
					int spaceWidth = gc.textExtent(" ").x;
					int tabWidth = textWithTabs.tabs * widget.getTabs() * spaceWidth;
					int ghostTextWidth = tabWidth + gc.stringExtent(textWithTabs.text).x;
					Point location = widget.getLocationAtOffset(offset);
					gc.drawString(textWithTabs.text, location.x + tabWidth, location.y);

					// Leave the space for the ghost text
					setStyle(gc, originStyleRange);
					int shiftWidth = ghostTextWidth - gc.stringExtent(replacedText).x;
					int targetCharWidth = gc.stringExtent(targetChar).x;

					StyleRange currentStyleRange = widget.getStyleRangeAtOffset(targetOffset);
					if (currentStyleRange != null && currentStyleRange.metrics != null
							&& currentStyleRange.metrics.width == shiftWidth + targetCharWidth) {
						// nothing to do
					} else {
						StyleRange styleRange = (StyleRange) originStyleRange.clone();
						styleRange.start = targetOffset;
						styleRange.length = 1;
						FontMetrics fontMetrics = gc.getFontMetrics();
						GlyphMetrics glyphMetrics = new GlyphMetrics(fontMetrics.getAscent(), fontMetrics.getDescent(),
								shiftWidth + targetCharWidth);
						modifiedGlyphMetrics.add(glyphMetrics);
						styleRange.metrics = glyphMetrics;
						widget.setStyleRange(styleRange);
						logger.debug("Set StyleRange:" + styleRange.start + " -> " + styleRange.metrics);
					}

					// Draw the moved char
					Point targetCharLocation = widget.getLocationAtOffset(targetOffset);
					gc.drawString(targetChar, targetCharLocation.x + shiftWidth, targetCharLocation.y, true);
				});
			}
		}

		private void drawSuffixLines(int offset, String text) {
			logger.debug("drawSuffixLines:" + offset + ":" + text);
			StyledText widget = getWidget();
			int lineHeight = widget.getLineHeight();
			List<String> lines = text.lines().toList();

			// Leave the space for the ghost text
			int nextLine = widget.getLineAtOffset(offset) + 1;
			if (nextLine < widget.getLineCount()) {
				int lineCount = lines.size();
				int originVerticalIndent = widget.getLineVerticalIndent(nextLine);
				Position position = new Position(widget.getOffsetAtLine(nextLine), 0);
				positionManager.managePosition(position);
				int modifiedVerticalIndent = originVerticalIndent + lineCount * lineHeight;
				modifiedLinesVerticalIndent
						.add(new ModifiedLineVerticalIndent(position, originVerticalIndent, modifiedVerticalIndent));
				widget.setLineVerticalIndent(nextLine, modifiedVerticalIndent);
				logger.debug("Set LineVerticalIndent:" + nextLine + " -> " + modifiedVerticalIndent);
			}

			List<TextWithTabs> linesTextWithTab = new ArrayList<>();
			for (String line : lines) {
				linesTextWithTab.add(splitLeadingTabs(line));
			}

			paintFunctions.add((gc) -> {
				// Draw ghost text
				setStyleToGhostText(gc);
				int spaceWidth = gc.textExtent(" ").x;
				Point location = widget.getLocationAtOffset(offset);
				int y = location.y;
				for (TextWithTabs textWithTabs : linesTextWithTab) {
					int x = widget.getLeftMargin() + textWithTabs.tabs * widget.getTabs() * spaceWidth;
					y += lineHeight;
					gc.drawString(textWithTabs.text, x, y, true);
				}
			});
		}

		private void setStyle(GC gc, StyleRange styleRange) {
			if (styleRange.foreground != null) {
				gc.setForeground(styleRange.foreground);
			} else {
				gc.setForeground(getWidget().getForeground());
			}
			if (styleRange.font != null) {
				gc.setFont(styleRange.font);
			} else {
				gc.setFont(getWidget().getFont());
			}
		}

		private void setStyleToGhostText(GC gc) {
			gc.setForeground(getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
			if (font == null || font.isDisposed()) {
				FontData[] fontData = getWidget().getFont().getFontData();
				for (int i = 0; i < fontData.length; ++i) {
					fontData[i].setStyle(fontData[i].getStyle() | SWT.ITALIC);
				}
				font = new Font(getDisplay(), fontData);
			}
			gc.setFont(font);
		}

		static final Pattern PATTERN_LEADING_TABS = Pattern.compile("^(\\t*)(.*)$");

		private static TextWithTabs splitLeadingTabs(String text) {
			Matcher matcher = PATTERN_LEADING_TABS.matcher(text);
			if (matcher.matches()) {
				return new TextWithTabs(matcher.group(1).length(), matcher.group(2));
			} else {
				return new TextWithTabs(0, text);
			}
		}

		private static class ModifiedLineVerticalIndent {
			private Position position;
			private int indent;
			private int modifiedIndent;

			public ModifiedLineVerticalIndent(Position position, int indent, int modifiedIndent) {
				this.position = position;
				this.indent = indent;
				this.modifiedIndent = modifiedIndent;
			}
		}

		private static class TextWithTabs {
			private int tabs;
			private String text;

			public TextWithTabs(int tabs, String text) {
				this.tabs = tabs;
				this.text = text;
			}
		}
	}
}
