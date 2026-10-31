export type BootstrapPhase =
  | "disabled"
  | "pending"
  | "downloading"
  | "ready"
  | "failed";

export interface BootstrapState {
  phase: BootstrapPhase;
  message: string;
  ready: boolean;
  /** 失败原因，phase 为 failed 时存在 */
  reason?: string;
}

export type BootstrapListener = (state: BootstrapState) => void;

/**
 * 状态机：缓存最近一次状态并向订阅者广播，
 * 让后到的插件也能立刻拿到当前进度。
 */
export class BootstrapStateStore {
  private state: BootstrapState;
  private readonly listeners = new Set<BootstrapListener>();

  constructor(initial: BootstrapState) {
    this.state = initial;
  }

  get current(): BootstrapState {
    return this.state;
  }

  set(next: BootstrapState): void {
    this.state = next;
    for (const listener of this.listeners) {
      try {
        listener(next);
      } catch {
        // 订阅者异常不应中断下载流程
      }
    }
  }

  subscribe(listener: BootstrapListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  }
}
