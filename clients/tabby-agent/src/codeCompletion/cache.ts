import { LRUCache } from "lru-cache";
import { CompletionSolution, CompletionItem } from "./solution";
import { getLogger } from "../logger";

export class CompletionCache {
  private readonly logger = getLogger("CompletionCache");
  private readonly config = {
    maxCount: 1000,
    forwardCachingChars: 50,
  };
  private cache = new LRUCache<string, CompletionSolution>({
    max: this.config.maxCount,
  });

  has(key: string): boolean {
    return this.cache.has(key);
  }

  update(value: CompletionSolution): void {
    this.logger.debug(`Updating completion cache, cache number before updating: ${this.cache.size}`);
    const solutions = [value, ...this.generateForwardSolutions(value)];
    solutions.forEach((solution) => {
      const cachedSolution = this.cache.get(solution.context.hash);
      if (cachedSolution) {
        this.cache.set(solution.context.hash, CompletionSolution.merge(cachedSolution, solution));
      } else {
        this.cache.set(solution.context.hash, solution);
      }
    });
    this.logger.debug(`Updated entries number: ${solutions.length}`);
    this.logger.debug(`Completion cache updated, cache number: ${this.cache.size}`);
  }

  get(key: string): CompletionSolution | undefined {
    return this.cache.get(key);
  }

  private generateForwardSolutions(solution: CompletionSolution): CompletionSolution[] {
    const forwardSolutions: CompletionSolution[] = [];
    const pushForwardSolution = (item: CompletionItem) => {
      const existSolution = forwardSolutions.find((solution) => solution.context.hash === item.context.hash);
      if (existSolution) {
        existSolution.add(item);
      } else {
        forwardSolutions.push(new CompletionSolution(item.context, [item]));
      }
    };
    for (const item of solution.items) {
      // Forward at current line
      for (let chars = 1; chars < Math.min(this.config.forwardCachingChars, item.currentLine.length); chars++) {
        pushForwardSolution(item.forward(chars));
      }
      if (item.lines.length > 2) {
        // current line end
        pushForwardSolution(item.forward(item.currentLine.length - 1));
        // next line start
        pushForwardSolution(item.forward(item.currentLine.length));
        const nextLine = item.lines[1]!;
        let spaces = nextLine.search(/\S/);
        if (spaces < 0) {
          spaces = nextLine.length - 1;
        }
        // next line start, after indent spaces
        pushForwardSolution(item.forward(item.currentLine.length + spaces));
      }
    }
    return forwardSolutions;
  }
}
