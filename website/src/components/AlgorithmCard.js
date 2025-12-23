// website/src/components/AlgorithmCard.js
import React, { useState } from 'react';

export default function AlgorithmCard({ name, badges = [], short, complexity, params = {}, exampleCode = '', pros = [], cons = [] }) {
  const [open, setOpen] = useState(false);
  return (
    <article className="ifx-algo-card" role="article" aria-labelledby={`alg-${name.replace(/\s+/g,'-')}`}>
      <div className="title" id={`alg-${name.replace(/\s+/g,'-')}`}>
        <strong>{name}</strong>
        <div style={{marginLeft:'8px'}} className="badges">
          {badges.map((b,i)=> <span key={i} className="badge">{b}</span>)}
        </div>
      </div>
      <div style={{marginTop:8,color:'#cbd5e1'}}>{short}</div>
      <div style={{marginTop:8, display:'flex', gap:12, alignItems:'center', color:'#9aa6b2', fontSize:'.95rem'}}>
        <div><strong>Complexity:</strong> <span style={{color:'#e6eef6'}}>{complexity}</span></div>
        <div><strong>Params:</strong> <span style={{color:'#e6eef6'}}>{Object.keys(params).slice(0,3).map(k=> `${k}=${params[k]}`).join(', ')}</span></div>
      </div>
      <div style={{marginTop:8, display:'flex', gap:12}}>
        <div style={{flex:1}}>
          <strong>Pros</strong>
          <ul>{pros.map((p,i)=> <li key={i} style={{color:'#cbd5e1'}}>{p}</li>)}</ul>
        </div>
        <div style={{flex:1}}>
          <strong>Cons</strong>
          <ul>{cons.map((c,i)=> <li key={i} style={{color:'#cbd5e1'}}>{c}</li>)}</ul>
        </div>
      </div>
      <div style={{marginTop:12}}>
        <button onClick={()=>setOpen(!open)} aria-expanded={open} style={{background:'transparent',border:'1px solid rgba(255,255,255,0.06)', color:'#e6eef6', padding:'6px 10px', borderRadius:8}}>
          {open ? 'Hide example' : 'Show example'}
        </button>
        {open && (
          <div style={{marginTop:12}}>
            <pre className="prism-code"><code>{exampleCode}</code></pre>
          </div>
        )}
      </div>
    </article>
  );
}
// website/src/components/AlgorithmCard.js
import React, { useState } from 'react';

export default function AlgorithmCard({ name, badges = [], short, complexity, params = {}, exampleCode = '', pros = [], cons = [] }) {
  const [open, setOpen] = useState(false);
  return (
    <article className="ifx-algo-card" role="article" aria-labelledby={`alg-${name.replace(/\s+/g,'-')}`}>
      <div className="title" id={`alg-${name.replace(/\s+/g,'-')}`}>
        <strong>{name}</strong>
        <div style={{marginLeft:'8px'}} className="badges">
          {badges.map((b,i)=> <span key={i} className="badge">{b}</span>)}
        </div>
      </div>
      <div style={{marginTop:8,color:'#cbd5e1'}}>{short}</div>
      <div style={{marginTop:8, display:'flex', gap:12, alignItems:'center', color:'#9aa6b2', fontSize:'.95rem'}}>
        <div><strong>Complexity:</strong> <span style={{color:'#e6eef6'}}>{complexity}</span></div>
        <div><strong>Params:</strong> <span style={{color:'#e6eef6'}}>{Object.keys(params).slice(0,3).map(k=> `${k}=${params[k]}`).join(', ')}</span></div>
      </div>
      <div style={{marginTop:8, display:'flex', gap:12}}>
        <div style={{flex:1}}>
          <strong>Pros</strong>
          <ul>{pros.map((p,i)=> <li key={i} style={{color:'#cbd5e1'}}>{p}</li>)}</ul>
        </div>
        <div style={{flex:1}}>
          <strong>Cons</strong>
          <ul>{cons.map((c,i)=> <li key={i} style={{color:'#cbd5e1'}}>{c}</li>)}</ul>
        </div>
      </div>
      <div style={{marginTop:12}}>
        <button onClick={()=>setOpen(!open)} aria-expanded={open} style={{background:'transparent',border:'1px solid rgba(255,255,255,0.06)', color:'#e6eef6', padding:'6px 10px', borderRadius:8}}>
          {open ? 'Hide example' : 'Show example'}
        </button>
        {open && (
          <div style={{marginTop:12}}>
            <pre className="prism-code"><code>{exampleCode}</code></pre>
          </div>
        )}
      </div>
    </article>
  );
}
