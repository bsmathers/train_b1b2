"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Result = exports.Choice = void 0;

class Choice {
  static Encoding = { pass: 0, move: 1, switch: 2 };
  static Types = Object.keys(Choice.Encoding);
  static MATCH = /^(?:(pass)|((move) ([0-4]))|((switch) ([2-6])))$/;

  constructor() {}

  static decode(byte) {
    return { type: Choice.Types[byte & 0b11], data: byte >> 2 };
  }

  static encode(choice) {
    return choice ? (choice.data << 2 | Choice.Encoding[choice.type]) : 0;
  }

  static parse(choice) {
    const m = Choice.MATCH.exec(choice);
    if (!m) throw new Error(`Invalid choice: '${choice}'`);
    const type = (m[1] ?? m[3] ?? m[6]);
    const data = +(m[4] ?? m[7] ?? 0);
    return { type, data };
  }

  static format(choice) {
    return choice.type === 'pass' ? choice.type : `${choice.type} ${choice.data}`;
  }

  static pass = { type: 'pass', data: 0 };

  static move(data) {
    return { type: 'move', data };
  }

  static switch(data) {
    return { type: 'switch', data };
  }
}
exports.Choice = Choice;

class Result {
  static Encoding = { win: 1, lose: 2, tie: 3, error: 4 };
  static Types = [undefined, 'win', 'lose', 'tie', 'error'];

  constructor() {}

  static decode(byte) {
    return {
      type: Result.Types[byte & 0b1111],
      p1: Choice.Types[(byte >> 4) & 0b11],
      p2: Choice.Types[byte >> 6],
    };
  }

  static encode(result) {
    return (result.type ? Result.Encoding[result.type] : 0) |
      (Choice.Encoding[result.p1] << 4) |
      (Choice.Encoding[result.p2] << 6);
  }
}
exports.Result = Result;
