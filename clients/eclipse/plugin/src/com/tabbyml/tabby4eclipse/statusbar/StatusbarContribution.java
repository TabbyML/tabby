package com.tabbyml.tabby4eclipse.statusbar;

import org.eclipse.lsp4j.Command;
import org.eclipse.lsp4j.ExecuteCommandParams;
import org.eclipse.swt.custom.CLabel;
import org.eclipse.swt.events.MouseAdapter;
import org.eclipse.swt.events.MouseEvent;
import org.eclipse.swt.events.SelectionAdapter;
import org.eclipse.swt.events.SelectionEvent;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Control;
import org.eclipse.swt.widgets.Menu;
import org.eclipse.swt.widgets.MenuItem;
import org.eclipse.ui.menus.WorkbenchWindowControlContribution;

import com.tabbyml.tabby4eclipse.Images;
import com.tabbyml.tabby4eclipse.lsp.LanguageServerService;
import com.tabbyml.tabby4eclipse.lsp.protocol.StatusInfo;

public class StatusbarContribution extends WorkbenchWindowControlContribution {
	private static final String TOOLTIP_INITIALIZATION_FAILED = "Tabby: Initialization Failed";

	private StatusInfoHolder statusInfoHolder = StatusInfoHolder.getInstance();

	@Override
	protected Control createControl(Composite parent) {
		CLabel label = new CLabel(parent, 0);
		label.setText("Tabby");
		label.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseUp(MouseEvent e) {
				Menu menu = createMenu(label);
				menu.setVisible(true);
			}
		});

		updateLabel(label);

		StatusInfoHolder.getInstance().addStatusDidChangeListener(() -> {
			label.getDisplay().asyncExec(() -> {
				updateLabel(label);
			});
		});

		return label;
	}

	private void updateLabel(CLabel label) {
		if (statusInfoHolder.isConnectionFailed()) {
			label.setImage(Images.getIcon(Images.ICON_ERROR));
			label.setToolTipText(TOOLTIP_INITIALIZATION_FAILED);
		}
		StatusInfo statusInfo = statusInfoHolder.getStatusInfo();
		if (statusInfo.getTooltip() != null) {
			label.setToolTipText(statusInfo.getTooltip());
		} else {
			label.setToolTipText("Tabby: " + statusInfo.getStatus());
		}
		switch (statusInfo.getStatus()) {
		case StatusInfo.Status.NOT_INITIALIZED:
			label.setImage(Images.getIcon(Images.ICON_LOADING));
			break;
		case StatusInfo.Status.CONNECTING:
			label.setImage(Images.getIcon(Images.ICON_LOADING));
			break;
		case StatusInfo.Status.UNAUTHORIZED:
			label.setImage(Images.getIcon(Images.ICON_WARN));
			break;
		case StatusInfo.Status.DISCONNECTED:
			label.setImage(Images.getIcon(Images.ICON_ERROR));
			break;
		case StatusInfo.Status.READY:
			label.setImage(Images.getIcon(Images.ICON_CHECK));
			break;
		case StatusInfo.Status.FETCHING:
			label.setImage(Images.getIcon(Images.ICON_LOADING));
			break;
		case StatusInfo.Status.COMPLETION_RESPONSE_SLOW:
			label.setImage(Images.getIcon(Images.ICON_WARN));
			break;
		default:
			break;
		}
	}

	private Menu createMenu(CLabel label) {
		Menu menu = new Menu(label);
		MenuItem statusItem = new MenuItem(menu, 0);

		if (statusInfoHolder.isConnectionFailed()) {
			statusItem.setImage(Images.getIcon(Images.ICON_ERROR));
			statusItem.setText(TOOLTIP_INITIALIZATION_FAILED);
		}

		StatusInfo statusInfo = statusInfoHolder.getStatusInfo();
		if (statusInfo.getTooltip() != null) {
			statusItem.setText(statusInfo.getTooltip());
		} else {
			statusItem.setText("Tabby: " + statusInfo.getStatus());
		}
		Command commmand = statusInfo.getCommand();
		if (commmand != null) {
			statusItem.addSelectionListener(new SelectionAdapter() {
				@Override
				public void widgetSelected(SelectionEvent e) {
					LanguageServerService.getInstance().getServer().execute((server) -> {
						ExecuteCommandParams params = new ExecuteCommandParams();
						params.setCommand(commmand.getCommand());
						params.setArguments(commmand.getArguments());
						server.getWorkspaceService().executeCommand(params);
						return null;
					});
				}
			});
		}
		switch (statusInfo.getStatus()) {
		case StatusInfo.Status.NOT_INITIALIZED:
			statusItem.setImage(Images.getIcon(Images.ICON_LOADING));
			break;
		case StatusInfo.Status.CONNECTING:
			statusItem.setImage(Images.getIcon(Images.ICON_LOADING));
			break;
		case StatusInfo.Status.UNAUTHORIZED:
			statusItem.setImage(Images.getIcon(Images.ICON_WARN));
			break;
		case StatusInfo.Status.DISCONNECTED:
			statusItem.setImage(Images.getIcon(Images.ICON_ERROR));
			break;
		case StatusInfo.Status.READY:
			statusItem.setImage(Images.getIcon(Images.ICON_CHECK));
			break;
		case StatusInfo.Status.FETCHING:
			statusItem.setImage(Images.getIcon(Images.ICON_LOADING));
			break;
		case StatusInfo.Status.COMPLETION_RESPONSE_SLOW:
			statusItem.setImage(Images.getIcon(Images.ICON_WARN));
			break;
		default:
			break;
		}
		return menu;
	}
}
