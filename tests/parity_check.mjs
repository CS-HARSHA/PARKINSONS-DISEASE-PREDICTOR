import fs from 'node:fs';
const {macd}=await import('../webapp/js/macd.js');
const cases=JSON.parse(fs.readFileSync(new URL('./fixtures/macd_parity.json',import.meta.url)));
for(const c of cases){const got=macd(c.h,c.ql,c.qh);if(c.expected===null){if(Number.isFinite(got))throw new Error(`expected NaN, got ${got}`)}else if(!Number.isFinite(got)||Math.abs(got-c.expected)>1e-9)throw new Error(`mismatch: ${got} vs ${c.expected}`)}
console.log(`passed ${cases.length} MACD parity cases`);
