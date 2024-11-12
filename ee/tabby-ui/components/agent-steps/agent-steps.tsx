import React from "react";
import * as Accordion from "@radix-ui/react-accordion";
import { ChevronDownIcon } from "@radix-ui/react-icons";
import "./styles.css";
import { cn } from '@/lib/utils'
import { IconAgent, IconResult, IconSearch } from '@/components/ui/icons'

export const AgentSteps = () => (
	<Accordion.Root
		className="AccordionRoot"
		type="single"
		defaultValue="item-1"
		collapsible
	>
		<Accordion.Item className="AccordionItem" value="item-1">
			<AccordionTrigger>
				<div className="AgentHeader">
					<div className="AgentName">Tabby Agent</div>
					<div className="StepCounts">3-steps</div>
				</div>
			</AccordionTrigger>
			<AccordionContent>
				<div className="step">
					<div className="step-title text-sm font-bold leading-none flex items-center">
						<span className="inline-flex px-2">
							<IconAgent />
						</span>
						
						<span>Thinking</span></div>
					<div className="step-content p-2">task</div>
				</div>
				<div className="step">
					<div className="step-title text-sm font-bold leading-none flex items-center">
						<span className="inline-flex px-2"><IconResult /></span>
						Searching web</div>
					<div className="step-content p-2">task</div>
				</div>
				<div className="step">
					<div className="step-title text-sm font-bold leading-none flex items-center">
						<span className="inline-flex px-2"><IconSearch /></span>
						Combining results</div>
					<div className="step-content p-2">task</div>
				</div>
			</AccordionContent>
		</Accordion.Item>
	</Accordion.Root>
);

const AccordionTrigger = React.forwardRef<
	React.ElementRef<typeof Accordion.Trigger>,
	React.ComponentPropsWithoutRef<typeof Accordion.Trigger>
	>(
	({ children, className, ...props }, forwardedRef) => (
		<Accordion.Header className="AccordionHeader">
			<Accordion.Trigger
				className={cn("AccordionTrigger", className)}
				{...props}
				ref={forwardedRef}
			>
				{children}
				<ChevronDownIcon className="AccordionChevron" aria-hidden />
			</Accordion.Trigger>
		</Accordion.Header>
	),
);

const AccordionContent = React.forwardRef<
	React.ElementRef<typeof Accordion.Content>,
	React.ComponentPropsWithoutRef<typeof Accordion.Content>
>(
	({ children, className, ...props }, forwardedRef) => (
		<Accordion.Content
			className={cn("AccordionContent", className)}
			{...props}
			ref={forwardedRef}
		>
			<div className="AccordionContentText">{children}</div>
		</Accordion.Content>
	),
);
