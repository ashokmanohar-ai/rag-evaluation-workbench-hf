import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
env.allowRemoteModels = true;
env.allowLocalModels = false;
env.useBrowserCache = true;

const DEFAULT_CORPUS = `doc_id,text
FHIR-01,"A FHIR CapabilityStatement describes server capabilities including FHIR version, supported resources, interactions, formats, search parameters and operations. Quality engineering can compare declared capability with live behavior."
FHIR-02,"FHIR resource validation can use the $validate operation. Servers return an OperationOutcome containing issues with severity, code, diagnostics and locations. Terminology validation can use $validate-code against a CodeSystem or ValueSet."
MCP-01,"Enterprise MCP governance should classify tool actions and enforce outcomes such as ALLOW, APPROVAL, BLOCK, RETRY and ESCALATE. Destructive or cross-tenant operations should be blocked or require explicit approval."
AGENT-01,"Agent evaluation should measure task success, tool correctness, safety, latency and flakiness over repeated trials. Hard release gates prevent unsafe or unreliable agent versions from reaching production."
RAG-01,"Retrieval evaluation measures whether relevant evidence appears in the ranked result set. Common information retrieval metrics include Precision at K, Recall at K, Hit Rate, Mean Reciprocal Rank and normalized Discounted Cumulative Gain."
RAG-02,"BM25 is a lexical ranking function based on term frequency, inverse document frequency and document length normalization. The k1 parameter controls term saturation while b controls length normalization."
RAG-03,"Semantic retrieval encodes queries and passages into vectors and ranks by cosine similarity. Hybrid retrieval combines lexical BM25 evidence with semantic similarity to improve robustness across exact terminology and paraphrases."
AIQX-01,"AIQX aggregates evidence from agent evaluation, AgentOps, RAGOps and healthcare interoperability checks. A weighted release gate can produce PASS, CONDITIONAL GO or HOLD while retaining source provenance, timestamps and evidence hashes."
FDE-01,"A forward deployed engineer translates customer requirements into architecture alternatives, trade-offs, risks, assumptions, proof-of-concept exit criteria and an architecture decision record."
VERCEL-01,"A production delivery path can use source control, pull requests, preview deployments, automated checks and production promotion. Observability should include latency, errors, deployment health and evidence provenance."`;
const DEFAULT_EVAL = `query,relevant_doc_ids
How do I verify what a FHIR server supports?,FHIR-01
What operation returns validation issues for a Patient resource?,FHIR-02
Which MCP outcome should protect destructive actions?,MCP-01
How should repeated AI agent trials be evaluated?,AGENT-01
Which metrics show ranked retrieval quality?,RAG-01
What do k1 and b mean in BM25?,RAG-02
How does hybrid retrieval combine exact terms and semantic similarity?,RAG-02|RAG-03
How can multiple AI quality signals become a release decision?,AIQX-01
What should an FDE produce from a customer requirement?,FDE-01
How should a production deployment workflow be structured?,VERCEL-01`;

const $ = id => document.getElementById(id);
$('corpus').value = DEFAULT_CORPUS;
$('eval').value = DEFAULT_EVAL;
let lastEvidence = null, extractor = null, extractorModel = null;

document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.pane').forEach(x=>x.classList.remove('active'));
  $(btn.dataset.pane).classList.add('active');
}));

document.querySelectorAll('.side-link').forEach(link => link.addEventListener('click', () => {
  if (!link.dataset.target) return;
  document.querySelectorAll('.side-link').forEach(x=>x.classList.remove('active'));
  link.classList.add('active');
  const target = link.dataset.target;
  if (['perquery','evidence','guide','architecture'].includes(target)) {
    const tab = document.querySelector(`.tab[data-pane="${target}"]`);
    if (tab) tab.click();
    document.querySelector('.results-shell')?.scrollIntoView({behavior:'smooth', block:'start'});
  } else {
    document.getElementById(target)?.scrollIntoView({behavior:'smooth', block:'start'});
  }
}));

function setStatus(text, cls=''){ const el=$('status'); el.textContent=text; el.className='status '+cls; }
function esc(v){ return String(v??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

function parseCSV(text){
  const rows=[]; let row=[], cell='', q=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){ if(q && text[i+1]==='"'){cell+='"';i++;} else q=!q; }
    else if(c===','&&!q){row.push(cell);cell='';}
    else if((c==='\n'||c==='\r')&&!q){ if(c==='\r'&&text[i+1]==='\n')i++; row.push(cell); if(row.some(x=>x.trim())) rows.push(row); row=[];cell=''; }
    else cell+=c;
  }
  row.push(cell); if(row.some(x=>x.trim())) rows.push(row);
  if(rows.length<2) throw new Error('CSV requires a header and at least one data row.');
  const headers=rows[0].map(x=>x.trim());
  return rows.slice(1).map(r=>Object.fromEntries(headers.map((h,i)=>[h,(r[i]??'').trim()])));
}
function tokens(text){ return (text.toLowerCase().match(/[a-z0-9_.$-]+/g)||[]); }
function chunkDocs(docs,size,overlap){
  if(overlap>=size) throw new Error('Overlap must be smaller than chunk size.');
  const out=[],step=size-overlap;
  docs.forEach(d=>{const words=d.text.split(/\s+/).filter(Boolean); for(let s=0,i=1;s<words.length;s+=step,i++){const part=words.slice(s,s+size);if(!part.length)break;out.push({chunk_id:`${d.doc_id}::c${i}`,doc_id:d.doc_id,text:part.join(' ')});if(s+size>=words.length)break;}}); return out;
}
function bm25Index(chunks,k1,b){
  const ts=chunks.map(c=>tokens(c.text)), lens=ts.map(x=>x.length), avg=lens.reduce((a,b)=>a+b,0)/Math.max(lens.length,1), df={};
  ts.forEach(arr=>[...new Set(arr)].forEach(t=>df[t]=(df[t]||0)+1));
  return query=>{const q=tokens(query),n=ts.length;return ts.map((arr,i)=>{const tf={};arr.forEach(t=>tf[t]=(tf[t]||0)+1);let score=0;q.forEach(term=>{const nq=df[term]||0;if(!nq)return;const idf=Math.log(1+(n-nq+.5)/(nq+.5)),f=tf[term]||0,den=f+k1*(1-b+b*lens[i]/Math.max(avg,1e-9));if(den)score+=idf*((f*(k1+1))/den);});return score;});};
}
function minmax(a){const lo=Math.min(...a),hi=Math.max(...a);if(Math.abs(hi-lo)<1e-12)return hi>0?a.map(()=>1):a.map(()=>0);return a.map(x=>(x-lo)/(hi-lo));}
function rank(chunks,scores){return chunks.map((c,i)=>({...c,score:+scores[i]})).sort((a,b)=>b.score-a.score);}
function uniqueDocs(ranked){const s=new Set(),o=[];ranked.forEach(r=>{if(!s.has(r.doc_id)){s.add(r.doc_id);o.push(r.doc_id);}});return o;}
function metrics(docRank,relevant,k){const rs=new Set(relevant),top=docRank.slice(0,k),hits=top.filter(d=>rs.has(d));let rr=0;for(let i=0;i<docRank.length;i++){if(rs.has(docRank[i])){rr=1/(i+1);break;}}const dcg=top.reduce((a,d,i)=>a+(rs.has(d)?1/Math.log2(i+2):0),0),ideal=Math.min(rs.size,k),idcg=Array.from({length:ideal},(_,i)=>1/Math.log2(i+2)).reduce((a,b)=>a+b,0);return{precision_at_k:hits.length/Math.max(k,1),recall_at_k:hits.length/Math.max(rs.size,1),hit_rate:hits.length?1:0,rr,ndcg_at_k:idcg?dcg/idcg:0};}
function percentile(arr,p){const a=[...arr].sort((x,y)=>x-y);if(!a.length)return 0;const idx=(a.length-1)*p,lo=Math.floor(idx),hi=Math.ceil(idx);return lo===hi?a[lo]:a[lo]+(a[hi]-a[lo])*(idx-lo);}
async function sha256(obj){const data=new TextEncoder().encode(JSON.stringify(obj));const hash=await crypto.subtle.digest('SHA-256',data);return [...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,'0')).join('');}
function tensorRows(t){const data=Array.from(t.data), dims=t.dims||[];if(dims.length===1)return[data];const n=dims[0],d=data.length/n;return Array.from({length:n},(_,i)=>data.slice(i*d,(i+1)*d));}
function dot(a,b){let s=0;for(let i=0;i<a.length;i++)s+=a[i]*b[i];return s;}
async function getExtractor(model){if(extractor&&extractorModel===model)return extractor;if(extractor?.dispose)await extractor.dispose();setStatus('Loading Hugging Face embedding model in your browser…','work');extractor=await pipeline('feature-extraction',model,{dtype:'q8'});extractorModel=model;return extractor;}
async function embedBatch(model,texts){const ex=await getExtractor(model);const out=await ex(texts,{pooling:'mean',normalize:true});return tensorRows(out);}

function renderTable(target,rows,columns){if(!rows.length){$(target).innerHTML='<div class="muted" style="padding:16px">No rows.</div>';return;}$(target).innerHTML=`<table class="table"><thead><tr>${columns.map(c=>`<th>${esc(c.label)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${columns.map(c=>`<td>${c.render?c.render(r[c.key],r):esc(r[c.key])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;}
function renderMetrics(a){const vals=[['Recall@K',a.recall_at_k,'%',100],['Precision@K',a.precision_at_k,'%',100],['MRR',a.mrr,'',1],['nDCG@K',a.ndcg_at_k,'',1],['Hit Rate',a.hit_rate,'%',100],['P50 latency',a.p50_latency_ms,' ms',1]];$('metrics').innerHTML=vals.map(([n,v,s,m])=>`<div class="metric"><span>${n}</span><b>${m===100?(v*m).toFixed(1):v.toFixed(3)}${s}</b></div>`).join('');}

async function run(){
  const btn=$('run');btn.disabled=true;$('download').disabled=true;setStatus('Validating benchmark inputs…','work');
  try{
    const docs=parseCSV($('corpus').value); if(!docs.every(d=>d.doc_id&&d.text))throw new Error('Corpus needs doc_id and text columns.');
    const ev=parseCSV($('eval').value); if(!ev.every(r=>r.query&&r.relevant_doc_ids))throw new Error('Evaluation CSV needs query and relevant_doc_ids columns.');
    const topK=Math.max(1,+$('topk').value||3), chunkWords=+$('chunkWords').value||80, overlap=+$('overlap').value||0,k1=+$('k1').value||1.5,b=+$('b').value||.75,sw=+$('semanticWeight').value||.55,retriever=$('retriever').value,model=$('model').value.trim();
    const chunks=chunkDocs(docs,chunkWords,overlap), bm=bm25Index(chunks,k1,b), queries=ev.map(x=>x.query); let chunkEmb=null,queryEmb=null;
    if(retriever!=='BM25'){setStatus('Computing semantic embeddings locally in your browser…','work');chunkEmb=await embedBatch(model,chunks.map(c=>c.text));queryEmb=await embedBatch(model,queries);}
    const per=[],evidence=[],lat=[];
    for(let qi=0;qi<ev.length;qi++){
      const row=ev[qi],rel=row.relevant_doc_ids.replaceAll(';','|').split('|').map(x=>x.trim()).filter(Boolean),t0=performance.now(),bs=bm(row.query);let scores=bs;
      if(retriever!=='BM25'){const sem=chunkEmb.map(v=>dot(v,queryEmb[qi]));scores=retriever==='Semantic (MiniLM)'?sem:minmax(bs).map((x,i)=>(1-sw)*x+sw*minmax(sem)[i]);}
      const ranked=rank(chunks,scores),elapsed=performance.now()-t0;lat.push(elapsed);const dr=uniqueDocs(ranked),m=metrics(dr,rel,topK);
      per.push({query:row.query,relevant:rel.join(' | '),top_docs:dr.slice(0,topK).join(' → '),precision:m.precision_at_k,recall:m.recall_at_k,rr:m.rr,ndcg:m.ndcg_at_k,hit:m.hit_rate,latency_ms:elapsed});
      ranked.slice(0,Math.max(topK,5)).forEach((x,i)=>evidence.push({query:row.query,rank:i+1,doc_id:x.doc_id,chunk_id:x.chunk_id,score:x.score,relevant:rel.includes(x.doc_id)?'YES':'NO',text:x.text}));
    }
    const avg=k=>per.reduce((a,r)=>a+r[k],0)/per.length,agg={precision_at_k:avg('precision'),recall_at_k:avg('recall'),mrr:avg('rr'),ndcg_at_k:avg('ndcg'),hit_rate:avg('hit'),p50_latency_ms:percentile(lat,.5),p95_latency_ms:percentile(lat,.95),documents:docs.length,chunks:chunks.length,queries:ev.length};
    const runId='RAG-'+new Date().toISOString().replace(/[-:.]/g,'').replace('T','T').replace('Z','Z'),payload={schema_version:'1.0',run_id:runId,retriever,configuration:{top_k:topK,chunk_words:chunkWords,overlap_words:overlap,bm25_k1:k1,bm25_b:b,semantic_model:model,semantic_weight:sw},aggregate:agg,per_query:per};payload.evidence_sha256=await sha256(payload);lastEvidence=payload;
    renderMetrics(agg);$('meta').innerHTML=`<span><b>${esc(retriever)}</b></span><span>Run <code>${esc(runId)}</code></span><span>Evidence <code>${payload.evidence_sha256.slice(0,16)}</code></span><span>${docs.length} docs · ${chunks.length} chunks · ${ev.length} queries</span>`;
    renderTable('perQueryTable',per,[{key:'query',label:'Query'},{key:'relevant',label:'Relevant'},{key:'top_docs',label:'Top docs'},{key:'precision',label:`Precision@${topK}`,render:v=>v.toFixed(3)},{key:'recall',label:`Recall@${topK}`,render:v=>v.toFixed(3)},{key:'rr',label:'RR',render:v=>v.toFixed(3)},{key:'ndcg',label:`nDCG@${topK}`,render:v=>v.toFixed(3)},{key:'hit',label:'Hit'},{key:'latency_ms',label:'Latency ms',render:v=>v.toFixed(2)}]);
    renderTable('evidenceTable',evidence,[{key:'query',label:'Query'},{key:'rank',label:'Rank'},{key:'doc_id',label:'Doc'},{key:'chunk_id',label:'Chunk'},{key:'score',label:'Score',render:v=>v.toFixed(5)},{key:'relevant',label:'Relevant',render:v=>`<span class="${v==='YES'?'yes':'no'}">${v}</span>`},{key:'text',label:'Evidence text'}]);
    $('download').disabled=false;setStatus(`Completed ${retriever} benchmark. Evidence is reproducible and ready to export.`,'ok');
  }catch(e){console.error(e);setStatus(e?.message||String(e),'err');}finally{btn.disabled=false;}
}
$('run').addEventListener('click',run);
$('download').addEventListener('click',()=>{if(!lastEvidence)return;const blob=new Blob([JSON.stringify(lastEvidence,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${lastEvidence.run_id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
