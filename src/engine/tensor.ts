export function computeContiguousStrides(shape: number[]): number[] {
  const rank = shape.length;
  const strides = new Array<number>(rank);
  let stride = 1;
  for (let i = rank - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

export class Tensor {
  readonly data: Float32Array;
  readonly shape: number[];
  readonly strides: number[];
  readonly offset: number;

  constructor(
    data: Float32Array,
    shape: number[],
    strides?: number[],
    offset: number = 0
  ) {
    this.data = data;
    this.shape = [...shape];
    this.strides = strides ? [...strides] : computeContiguousStrides(shape);
    this.offset = offset;
  }

  static zeros(shape: number[]): Tensor {
    let size = 1;
    for (const d of shape) {
      size *= d;
    }
    return new Tensor(new Float32Array(size), shape);
  }

  static fromArray(values: number[], shape?: number[]): Tensor {
    const s = shape ? [...shape] : [values.length];
    return new Tensor(new Float32Array(values), s);
  }

  get size(): number {
    let s = 1;
    for (const d of this.shape) {
      s *= d;
    }
    return s;
  }

  get rank(): number {
    return this.shape.length;
  }

  slice(dim: number, start: number, end?: number): Tensor {
    if (dim < 0 || dim >= this.rank) {
      throw new Error(`Dimension ${dim} out of bounds for tensor of rank ${this.rank}`);
    }
    const dimSize = this.shape[dim];
    const actualEnd = end === undefined ? dimSize : end;

    if (start < 0 || actualEnd > dimSize || start > actualEnd) {
      throw new Error(
        `Slice range [${start}, ${actualEnd}) out of bounds for dim ${dim} with size ${dimSize}`
      );
    }

    const newShape = [...this.shape];
    newShape[dim] = actualEnd - start;

    const newOffset = this.offset + start * this.strides[dim];

    return new Tensor(this.data, newShape, this.strides, newOffset);
  }

  clone(): Tensor {
    const cloned = new Float32Array(this.size);
    if (this.isContiguous()) {
      cloned.set(this.data.subarray(this.offset, this.offset + this.size));
    } else {
      let dstIdx = 0;
      const copyRecursive = (dim: number, currOffset: number) => {
        if (dim === this.rank - 1) {
          const stride = this.strides[dim];
          const len = this.shape[dim];
          for (let i = 0; i < len; i++) {
            cloned[dstIdx++] = this.data[currOffset + i * stride];
          }
        } else {
          const stride = this.strides[dim];
          const len = this.shape[dim];
          for (let i = 0; i < len; i++) {
            copyRecursive(dim + 1, currOffset + i * stride);
          }
        }
      };
      copyRecursive(0, this.offset);
    }
    return new Tensor(cloned, this.shape);
  }

  reshape(newShape: number[]): Tensor {
    let newSize = 1;
    for (const d of newShape) {
      newSize *= d;
    }
    if (newSize !== this.size) {
      throw new Error(
        `Cannot reshape tensor of size ${this.size} to shape [${newShape.join(', ')}] with size ${newSize}`
      );
    }

    if (!this.isContiguous()) {
      return this.clone().reshape(newShape);
    }

    return new Tensor(this.data, newShape, computeContiguousStrides(newShape), this.offset);
  }

  isContiguous(): boolean {
    const expectedStrides = computeContiguousStrides(this.shape);
    for (let i = 0; i < this.rank; i++) {
      if (this.strides[i] !== expectedStrides[i]) {
        return false;
      }
    }
    return true;
  }

  toJSON(): { shape: number[]; data: number[] } {
    const arr: number[] = new Array(this.size);
    if (this.isContiguous()) {
      for (let i = 0; i < this.size; i++) {
        arr[i] = this.data[this.offset + i];
      }
    } else {
      let idx = 0;
      const readRecursive = (dim: number, currOffset: number) => {
        if (dim === this.rank - 1) {
          const stride = this.strides[dim];
          const len = this.shape[dim];
          for (let i = 0; i < len; i++) {
            arr[idx++] = this.data[currOffset + i * stride];
          }
        } else {
          const stride = this.strides[dim];
          const len = this.shape[dim];
          for (let i = 0; i < len; i++) {
            readRecursive(dim + 1, currOffset + i * stride);
          }
        }
      };
      readRecursive(0, this.offset);
    }
    return {
      shape: [...this.shape],
      data: arr,
    };
  }
}
