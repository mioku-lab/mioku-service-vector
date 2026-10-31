export class VectorCache {
  private readonly entries = new Map<string, Float32Array>();

  constructor(private readonly capacity: number) {}

  get(key: string): Float32Array | undefined {
    const hit = this.entries.get(key);
    if (!hit) return undefined;
    this.entries.delete(key);
    this.entries.set(key, hit);
    return hit;
  }

  set(key: string, vector: Float32Array): void {
    if (this.capacity <= 0) return;
    this.entries.delete(key);
    this.entries.set(key, vector);
    while (this.entries.size > this.capacity) {
      const oldest = this.entries.keys().next().value;
      if (oldest === undefined) break;
      this.entries.delete(oldest);
    }
  }

  clear(): void {
    this.entries.clear();
  }

  get size(): number {
    return this.entries.size;
  }
}
