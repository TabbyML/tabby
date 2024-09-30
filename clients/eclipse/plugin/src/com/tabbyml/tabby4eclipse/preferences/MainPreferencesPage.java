package com.tabbyml.tabby4eclipse.preferences;

import org.eclipse.jface.layout.GridDataFactory;
import org.eclipse.jface.preference.BooleanFieldEditor;
import org.eclipse.jface.preference.FieldEditorPreferencePage;
import org.eclipse.jface.preference.PreferenceDialog;
import org.eclipse.jface.preference.StringFieldEditor;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.Composite;
import org.eclipse.swt.widgets.Group;
import org.eclipse.swt.widgets.Label;
import org.eclipse.ui.IWorkbench;
import org.eclipse.ui.IWorkbenchPage;
import org.eclipse.ui.IWorkbenchPreferencePage;
import org.eclipse.ui.IWorkbenchWindow;
import org.eclipse.ui.PartInitException;
import org.eclipse.ui.PlatformUI;
import org.eclipse.ui.dialogs.PreferencesUtil;

import com.tabbyml.tabby4eclipse.Activator;

public class MainPreferencesPage extends FieldEditorPreferencePage implements IWorkbenchPreferencePage {
	public static final String ID = "com.tabbyml.tabby4eclipse.preferences.main";

	public static void openPreferences() {
		PreferenceDialog dialog = PreferencesUtil.createPreferenceDialogOn(
				PlatformUI.getWorkbench().getActiveWorkbenchWindow().getShell(), MainPreferencesPage.ID,
				new String[] { MainPreferencesPage.ID, "org.eclipse.lsp4e.preferences",
						"org.eclipse.lsp4e.logging.preferences", "org.eclipse.ui.preferencePages.Keys" },
				(Object) null);
		if (dialog != null) {
			dialog.open();
		}
	}

	public MainPreferencesPage() {
		super(GRID);
		setPreferenceStore(Activator.getDefault().getPreferenceStore());
	}

	@Override
	public void createFieldEditors() {
		Composite parent = getFieldEditorParent();
		createServerGroup(parent);
		createCompletionGroup(parent);
		createEnvironmentGroup(parent);
		createTelemetryGroup(parent);
	}

	private void createServerGroup(Composite parent) {
		Group group = new Group(parent, SWT.NONE);
		GridDataFactory.fillDefaults().align(SWT.FILL, SWT.FILL).grab(true, false).span(2, 1).applyTo(group);
		group.setText("Server");
		group.setLayout(new GridLayout());

		Composite grid = new Composite(group, SWT.NONE);
		grid.setLayout(new GridLayout(2, false));

		StringFieldEditor endpointInput = new StringFieldEditor(PreferencesService.KEY_SERVER_ENDPOINT, "Endpoint",
				grid);
		addField(endpointInput);
		StringFieldEditor tokenInput = new StringFieldEditor(PreferencesService.KEY_SERVER_TOKEN, "Token", grid);
		tokenInput.getTextControl(grid).setEchoChar('*');
		addField(tokenInput);

		Label tip = new Label(grid, SWT.WRAP);
		tip.setText(
				"Note: If leave empty, server endpoint config in `~/.tabby-client/agent/config.toml` will be used.");
		GridDataFactory.fillDefaults().indent(10, 2).span(2, 1).applyTo(tip);
	}

	private void createCompletionGroup(Composite parent) {
		Group group = new Group(parent, SWT.NONE);
		GridDataFactory.fillDefaults().align(SWT.FILL, SWT.FILL).grab(true, false).span(2, 1).applyTo(group);
		group.setText("Completion");
		group.setLayout(new GridLayout());

		Composite grid = new Composite(group, SWT.NONE);
		grid.setLayout(new GridLayout(2, false));

		BooleanFieldEditor autoTriggerCheckbox = new BooleanFieldEditor(
				PreferencesService.KEY_INLINE_COMPLETION_TRIGGER_AUTO, "Automatically trigger inline completion", grid);
		addField(autoTriggerCheckbox);
	}

	private void createEnvironmentGroup(Composite parent) {
		Group group = new Group(parent, SWT.NONE);
		GridDataFactory.fillDefaults().align(SWT.FILL, SWT.FILL).grab(true, false).span(2, 1).applyTo(group);
		group.setText("Environment");
		group.setLayout(new GridLayout());

		Composite grid = new Composite(group, SWT.NONE);
		grid.setLayout(new GridLayout(2, false));

		StringFieldEditor nodeBinaryPathInput = new StringFieldEditor(PreferencesService.KEY_NODE_BINARY_PATH,
				"Node.js binary path", grid);
		addField(nodeBinaryPathInput);

		Label tip = new Label(grid, SWT.WRAP);
		tip.setText(
				"Tabby will attempt to find the Node binary in the `PATH` environment variable for running tabby-agent.\nYou can specify the path to the Node.js binary here if required. Restart IDE to take effect.");
		GridDataFactory.fillDefaults().indent(10, 2).span(2, 1).applyTo(tip);
	}

	private void createTelemetryGroup(Composite parent) {
		Group group = new Group(parent, SWT.NONE);
		GridDataFactory.fillDefaults().align(SWT.FILL, SWT.FILL).grab(true, false).span(2, 1).applyTo(group);
		group.setText("Telemetry");
		group.setLayout(new GridLayout());

		Composite grid = new Composite(group, SWT.NONE);
		grid.setLayout(new GridLayout(2, false));

		BooleanFieldEditor disableAnonymousUsageTrackingCheckbox = new BooleanFieldEditor(
				PreferencesService.KEY_ANONYMOUS_USAGE_TRACKING_DISABLED, "Disable anonymous usage tracking", grid);
		addField(disableAnonymousUsageTrackingCheckbox);

		Label tip = new Label(grid, SWT.WRAP);
		tip.setText(
				"Tabby collects aggregated anonymous usage data and sends it to the Tabby team to help improve our products.\nYour code, generated completions, or any identifying information is never tracked or transmitted.\nFor more details on data collection, please check our online documentation.");
		GridDataFactory.fillDefaults().indent(10, 2).span(2, 1).applyTo(tip);
	}

	@Override
	public void init(IWorkbench workbench) {
	}
}
