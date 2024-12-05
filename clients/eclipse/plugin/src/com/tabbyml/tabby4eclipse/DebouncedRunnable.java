package com.tabbyml.tabby4eclipse;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class DebouncedRunnable {
	private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
	private ScheduledFuture<?> future;
	private final long delay;
	private final Runnable task;

	public DebouncedRunnable(Runnable task, long delay) {
		this.task = task;
		this.delay = delay;
	}

	public synchronized void call() {
		if (future != null && !future.isDone()) {
			future.cancel(true);
		}
		future = scheduler.schedule(task, delay, TimeUnit.MILLISECONDS);
	}

	// FIXME: scheduler shutdown not called
	public void shutdown() {
		scheduler.shutdown();
	}
}