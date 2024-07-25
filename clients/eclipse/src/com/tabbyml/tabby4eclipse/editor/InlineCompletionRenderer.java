package com.tabbyml.tabby4eclipse.editor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

	public void show(ITextViewer viewer, int offset, InlineCompletionItem completion) {
		if (currentTextViewer != null) {
			getPainter(currentTextViewer).update(null, 0);
		}
		currentTextViewer = viewer;
		currentCompletionItem = completion;
		getPainter(viewer).update(completion, offset);
	}
	
	public void hide() {
		if (currentTextViewer != null) {
			getPainter(currentTextViewer).update(null, 0);
			currentTextViewer = null;
			currentCompletionItem = null;
		}
	}
	
	public ITextViewer getCurrentTextViewer() {
		return currentTextViewer;
	}
	
	public InlineCompletionItem getCurrentCompletionItem() {
        return currentCompletionItem;
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
		private Map<Position, Integer> originLinesVerticalIndent = new HashMap<>();
		private List<StyleRange> originStyleRanges = new ArrayList<>();

		public InlineCompletionItemPainter(ITextViewer viewer) {
			this.viewer = viewer;
			getDisplay().syncExec(() -> {
				((ITextViewerExtension2) this.viewer).addPainter(this);
				getWidget().addPaintListener(this);
			});
		}

		public void update(InlineCompletionItem item, int offset) {
			if (this.item != item || this.offset != offset) {
				this.item = item;
				this.offset = offset;
				getWidget().redraw();
			}
		}

		@Override
		public void paintControl(PaintEvent event) {
			cleanup();
			render(event.gc);
		}

		@Override
		public void paint(int reason) {
			logger.debug("[Painter] paint called.");
		}

		@Override
		public void deactivate(boolean redraw) {
			logger.debug("[Painter] deactivate called.");
		}

		@Override
		public void dispose() {
			logger.debug("[Painter] dispose called.");
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
			originLinesVerticalIndent.forEach((position, indent) -> {
				int line = getWidget().getLineAtOffset(position.getOffset());
                getWidget().setLineVerticalIndent(line, indent);
                positionManager.unmanagePosition(position);
            });
			originLinesVerticalIndent.clear();
			originStyleRanges.forEach(range -> {
				getWidget().setStyleRange(range);
			});
			originStyleRanges.clear();
		}
		
		private void render(GC gc) {
			if (item == null) {
				return;
			}
			StyledText widget = getWidget();

			int prefixReplaceLength = offset - item.getReplaceRange().getStart();
			int suffixReplaceLength = item.getReplaceRange().getEnd() - offset;
			String text = item.getInsertText().substring(prefixReplaceLength);
			if (text.isEmpty()) {
			    return;
			}
			
			int currentLineEndOffset;
			int nextLineNumber = widget.getLineAtOffset(offset) + 1;
			if (nextLineNumber < widget.getLineCount()) {
				currentLineEndOffset = widget.getOffsetAtLine(widget.getLineAtOffset(offset) + 1) - 1;
			} else {
				currentLineEndOffset = widget.getCharCount() - 1;
			}
			String currentLineSuffix = "";
			if (offset < currentLineEndOffset && offset < widget.getCharCount()) {
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
	        		drawInsertPartText(gc, offset, textCurrentLine);
	        	} else {
	        		if (!textCurrentLine.isEmpty()) {
	        			drawCurrentLineText(gc, offset, textCurrentLine);
	        		}
	        		drawSuffixLines(gc, offset, textSuffixLines + currentLineSuffix);
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
			        	drawInsertPartText(gc, offset, insertPart);
			        }
					if (!appendPart.isEmpty()) {
						drawCurrentLineText(gc, offset + 1, appendPart);
					}
		        } else {
		        	drawInsertPartText(gc, offset, textCurrentLine, currentLineSuffix.substring(0, 1));
		        }
	        	if (!textSuffixLines.isEmpty()) {
	        		drawSuffixLines(gc, offset, textSuffixLines + currentLineSuffix.substring(1));
	        	}
	        } else {
	            // Replace range contains multiple chars
	        	if (textSuffixLines.isEmpty()) {
	        		drawInsertPartText(gc, offset, textCurrentLine, currentLineSuffix.substring(0, suffixReplaceLength));
	        	} else {
	        		if (!textCurrentLine.isEmpty()) {
	        			drawCurrentLineText(gc, offset, textCurrentLine);
	        		}
	        		drawSuffixLines(gc, offset, textSuffixLines + currentLineSuffix.substring(suffixReplaceLength));
	        	}
	        }
		}
		
		private void drawCurrentLineText(GC gc, int offset, String text) {
			StyledText widget = getWidget();

			// Draw ghost text
			setStyleToGhostText(gc);
			Point location = widget.getLocationAtOffset(offset);
			gc.drawString(text, location.x, location.y);
		}
		
		private void drawInsertPartText(GC gc, int offset, String text) {
			drawInsertPartText(gc, offset, text, "");
		}
		
		private void drawInsertPartText(GC gc, int offset, String text, String replaced) {
			StyledText widget = getWidget();

			// Leave the space for the ghost text
			int spaceWidth = gc.stringExtent(text).x - gc.stringExtent(replaced).x;
			int targetOffset = offset + replaced.length();
			String targetChar = widget.getText(targetOffset, targetOffset);
			int charWidth = gc.stringExtent(targetChar).x;
			StyleRange styleRange;
			StyleRange originStyleRange = widget.getStyleRangeAtOffset(targetOffset);
			if (originStyleRange == null) {
				originStyleRange = new StyleRange();
				originStyleRange.start = targetOffset;
				originStyleRange.length = 1;
			}
			originStyleRanges.add(originStyleRange);
			styleRange = (StyleRange) originStyleRange.clone();
			styleRange.start = targetOffset;
			styleRange.length = 1;
			FontMetrics fontMetrics = gc.getFontMetrics();
			GlyphMetrics glyphMetrics = new GlyphMetrics(fontMetrics.getAscent(), fontMetrics.getDescent(), spaceWidth + charWidth);
			styleRange.metrics = glyphMetrics;
			widget.setStyleRange(styleRange);

			// Draw the moved char
			setStyle(gc, styleRange);
			Point targetCharLocation = widget.getLocationAtOffset(targetOffset);
			gc.drawString(targetChar, targetCharLocation.x + spaceWidth, targetCharLocation.y, true);
			
			// Draw ghost text
			setStyleToGhostText(gc);
			Point location = widget.getLocationAtOffset(offset);
			gc.drawString(text, location.x, location.y);
		}
		
		private void drawSuffixLines(GC gc, int offset, String text) {
			StyledText widget = getWidget();
			int lineHeight = widget.getLineHeight();
			
			// Leave the space for the ghost text
			int nextLineNumber = widget.getLineAtOffset(offset) + 1;
			if (nextLineNumber < widget.getLineCount()) {
				int lineCount = (int) text.lines().count();
				int originVerticalIndent = widget.getLineVerticalIndent(nextLineNumber);
				Position position = new Position(widget.getOffsetAtLine(nextLineNumber), 0);
				originLinesVerticalIndent.put(position, originVerticalIndent);
                positionManager.managePosition(position);
				widget.setLineVerticalIndent(nextLineNumber, originVerticalIndent + lineHeight * lineCount);
			}
			
			// Draw ghost text
			setStyleToGhostText(gc);
			Point location = widget.getLocationAtOffset(offset);
			int x = widget.getLeftMargin();
			int y = location.y + lineHeight;
			gc.drawText(text, x, y, true);
		}

		private void setStyle(GC gc, StyleRange styleRange) {
			if (styleRange.foreground != null) {
				gc.setForeground(styleRange.foreground);
			} else {
				gc.setForeground(getWidget().getForeground());
			}
			if (styleRange.background != null) {
				gc.setBackground(styleRange.background);
			} else {
				gc.setBackground(getWidget().getBackground());
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
	}
}
