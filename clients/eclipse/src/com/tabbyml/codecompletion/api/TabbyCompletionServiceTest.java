package com.tabbyml.codecompletion.api;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class TabbyCompletionServiceTest {
	
	TabbyCompletionService tabbyService;
	
	@BeforeEach
	void setupTests() {
		tabbyService = new TabbyCompletionService("http://localhost:8080");
	}

	@Test
	void testInit() {
		assertEquals(tabbyService.getServerEndpoint(), "http://localhost:8080");
	}
	
	@Test
	void testExceptionOnNoHost() {
		tabbyService = new TabbyCompletionService("http://nirvana:8080");
		assertThrows(TabbyRequestException.class, () -> {
			tabbyService.completeWithTabby("prefix", "suffix");
        });
	}
	
	@Test
	void testExceptionOnMalformedUrl() {
		tabbyService = new TabbyCompletionService("localhost:8080");		
		assertThrows(TabbyRequestException.class, () -> {
			tabbyService.completeWithTabby("prefix", "suffix");
        });
	}
	
	@Test
	void testConnectionRefused() {
		tabbyService = new TabbyCompletionService("localhost:17777");		
		assertThrows(TabbyRequestException.class, () -> {
			tabbyService.completeWithTabby("prefix", "suffix");
        });
	}
	
	@Test
	void testHealthResponse() {
		TabbyV1HealthState response = tabbyService.getHealth();
		assertEquals(response.getClass(), TabbyV1HealthState.class);
	}

}
