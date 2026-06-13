// Sample TypeScript file for ingest testing

/** Greets a user by name. */
export function greetUser(name: string): string {
    return `Hello, ${name}!`;
}

/** A simple counter class. */
export class Counter {
    private count: number = 0;

    increment(): void {
        this.count++;
    }

    getValue(): number {
        return this.count;
    }
}

export interface Shape {
    area(): number;
    perimeter(): number;
}

export type Maybe<T> = T | null | undefined;

export enum Direction {
    Up = "UP",
    Down = "DOWN",
    Left = "LEFT",
    Right = "RIGHT",
}

export const MAX_RETRIES = 3;

// Non-exported: should NOT appear in the index
function internalHelper(x: number): number {
    return x * 2;
}

const internalConst = "secret";
