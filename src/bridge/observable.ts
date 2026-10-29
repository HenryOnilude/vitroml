export interface Observable<T> {
  subscribe(fn: (value: T) => void): () => void;
  get(): T;
}

export interface WritableObservable<T> extends Observable<T> {
  set(value: T): void;
  update(fn: (current: T) => T): void;
}

/**
 * Minimal observable store conforming to Svelte store contract:
 * { subscribe(run: (value: T) => void): () => void }
 * Zero UI/framework dependencies.
 */
export function createObservable<T>(initial: T): WritableObservable<T> {
  let value = initial;
  const subs = new Set<(v: T) => void>();

  return {
    subscribe(fn: (v: T) => void) {
      subs.add(fn);
      fn(value);
      return () => {
        subs.delete(fn);
      };
    },
    set(v: T) {
      value = v;
      for (const fn of subs) {
        fn(value);
      }
    },
    update(updater: (current: T) => T) {
      this.set(updater(value));
    },
    get() {
      return value;
    },
  };
}
