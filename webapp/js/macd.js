function quantileLinear(values,q){const a=values.slice().sort((x,y)=>x-y);if(!a.length)return NaN;const pos=(a.length-1)*q,lo=Math.floor(pos),hi=Math.ceil(pos);return lo===hi?a[lo]:a[lo]+(a[hi]-a[lo])*(pos-lo)}
export function macdWithCount(values,ql=.6,qh=1){const x=values.filter(Number.isFinite);if(x.length<2)return{value:NaN,pairs:0};const low=quantileLinear(x,ql),high=quantileLinear(x,qh);let sum=0,n=0;for(let i=0;i<x.length-1;i++){if(x[i]>=low&&x[i]<=high&&x[i+1]>=low&&x[i+1]<=high){sum+=Math.abs(x[i+1]-x[i]);n++}}return{value:n?sum/n:NaN,pairs:n}}
export function macd(values,ql=.6,qh=1){return macdWithCount(values,ql,qh).value}
export {quantileLinear as quantile};
