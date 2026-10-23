#!/usr/bin/env python3
"""Upload a server snapshot directly to GitHub Releases, without a local relay."""
import os,sys,json,time,hashlib,gzip,tarfile,threading,queue,fcntl,resource,stat,re,math,base64,http.client,urllib.request,urllib.error,urllib.parse,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/layer-probe')
WORK=Path('/root/autodl-tmp/.github_archive_20261023')
REPO='Liushengyu258/frozen-exit-adapters'
TAG='server-data-20261023'
CHUNK=512*1024**2
SMALL=64*1024**2
GUARD=threading.RLock()
HALT=threading.Event()
STATE={}
TOKEN=None
RELEASE=None
ASSETS={}
START=time.time()
PATTERN=re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{35,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{30,}')

def atomic(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(data,f,ensure_ascii=False,indent=2)
 os.replace(tmp,path)

def status(phase=None,**kw):
 with GUARD:
  if phase:STATE['phase']=phase
  STATE.update(kw);STATE['updated_epoch']=time.time();STATE['pid']=os.getpid()
  atomic(WORK/'status.json',STATE)

def log(message,**kw):
 print(json.dumps({'time':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'message':message,**kw},ensure_ascii=False),flush=True)

def api(method,path,data=None,attempts=6):
 url='https://api.github.com'+path
 for n in range(attempts):
  payload=None if data is None else json.dumps(data).encode()
  req=urllib.request.Request(url,data=payload,method=method,headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/vnd.github+json','Content-Type':'application/json','User-Agent':'layer-probe-server-archive','X-GitHub-Api-Version':'2022-11-28'})
  try:
   with urllib.request.urlopen(req,timeout=90) as response:
    content=response.read()
    return json.loads(content) if content else None
  except urllib.error.HTTPError as e:
   if e.code==404:raise
   if e.code in [429,500,502,503,504] or e.code==403 and e.headers.get('X-RateLimit-Remaining')=='0':
    if n+1<attempts:time.sleep(min(60,3*2**n));continue
   raise RuntimeError('GitHub API HTTP '+str(e.code)+' at '+path) from None
  except (OSError,TimeoutError):
   if n+1==attempts:raise
   time.sleep(min(60,3*2**n))
 raise RuntimeError('GitHub API retries exhausted')

def release_assets():
 out={};page=1
 while True:
  rows=api('GET',f'/repos/{REPO}/releases/{RELEASE["id"]}/assets?per_page=100&page={page}')
  for row in rows:out[row['name']]=row
  if len(rows)<100:break
  page+=1
 return out

def persist():
 with GUARD:atomic(WORK/'manifest.json',MANIFEST)

def digest_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while True:
   b=f.read(8*1024**2)
   if not b:break
   h.update(b)
 return h.hexdigest()

def confirm_asset(asset,size,checksum):
 if asset.get('size')!=size or asset.get('state')!='uploaded':return False
 digest=asset.get('digest')
 if digest:return digest=='sha256:'+checksum
 for _ in range(3):
  asset=api('GET',f'/repos/{REPO}/releases/assets/{asset["id"]}')
  digest=asset.get('digest')
  if digest:return digest=='sha256:'+checksum
  time.sleep(2)
 # Only the server reads back this data if GitHub cannot supply its SHA256.
 h=hashlib.sha256()
 req=urllib.request.Request(f'https://api.github.com/repos/{REPO}/releases/assets/{asset["id"]}',headers={'Authorization':'Bearer '+TOKEN,'Accept':'application/octet-stream','User-Agent':'layer-probe-server-archive'})
 class SafeRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,request,fp,code,msg,headers,newurl):
   out=super().redirect_request(request,fp,code,msg,headers,newurl)
   if out and urllib.parse.urlsplit(newurl).hostname!='api.github.com':out.remove_header('Authorization')
   return out
 with urllib.request.build_opener(SafeRedirect()).open(req,timeout=180) as response:
  while True:
   b=response.read(8*1024**2)
   if not b:break
   h.update(b)
 return h.hexdigest()==checksum

def send_asset(path,name):
 size=path.stat().st_size;checksum=digest_file(path)
 with GUARD:prior=ASSETS.get(name)
 if prior:
  if not confirm_asset(prior,size,checksum):raise RuntimeError('Existing asset does not match snapshot: '+name)
  return {'id':prior['id'],'name':name,'size':size,'sha256':checksum,'url':prior['browser_download_url'],'verified':True}
 for n in range(7):
  if HALT.is_set():raise RuntimeError('Upload halted')
  status(current_asset=name)
  conn=None
  try:
   conn=http.client.HTTPSConnection('uploads.github.com',timeout=180)
   endpoint=f'/repos/{REPO}/releases/{RELEASE["id"]}/assets?name='+urllib.parse.quote(name,safe='')
   conn.putrequest('POST',endpoint)
   for key,value in {'Authorization':'Bearer '+TOKEN,'User-Agent':'layer-probe-server-archive','Accept':'application/vnd.github+json','Content-Type':'application/octet-stream','Content-Length':str(size)}.items():conn.putheader(key,value)
   conn.endheaders();sent=0;last=time.time()
   with path.open('rb') as f:
    while True:
     b=f.read(4*1024**2)
     if not b:break
     conn.send(b);sent+=len(b)
     if time.time()-last>15:status(current_asset=name,current_asset_bytes_sent=sent,current_asset_bytes_total=size);last=time.time()
   response=conn.getresponse();payload=response.read()
   if response.status==201:
    row=json.loads(payload)
    if not confirm_asset(row,size,checksum):raise RuntimeError('Uploaded asset checksum mismatch: '+name)
    with GUARD:ASSETS[name]=row
    return {'id':row['id'],'name':name,'size':size,'sha256':checksum,'url':row['browser_download_url'],'verified':True}
   if response.status in [401,403]:raise RuntimeError('GitHub authorization failed with HTTP '+str(response.status))
   if response.status not in [408,422,429,500,502,503,504]:raise RuntimeError('Asset upload HTTP '+str(response.status))
  except (OSError,TimeoutError,http.client.HTTPException):pass
  finally:
   if conn:conn.close()
  # A disconnected upload may nevertheless have completed remotely.
  fresh=release_assets();row=fresh.get(name)
  if row:
   if row.get('state')=='uploaded' and confirm_asset(row,size,checksum):
    with GUARD:ASSETS[name]=row
    return {'id':row['id'],'name':name,'size':size,'sha256':checksum,'url':row['browser_download_url'],'verified':True}
   if row.get('size',0)==0 and row.get('state')!='uploaded':api('DELETE',f'/repos/{REPO}/releases/assets/{row["id"]}')
   else:raise RuntimeError('Incomplete or inconsistent release asset: '+name)
  if n==6:raise RuntimeError('Upload retries exhausted: '+name)
  time.sleep(min(60,3*2**n))
 raise RuntimeError('Asset upload failed: '+name)

def snapshot():
 entries=[];excluded=[]
 for path in sorted(ROOT.rglob('*')):
  rel=str(path.relative_to(ROOT));s=path.lstat();mode=stat.S_IMODE(s.st_mode)
  if path.is_symlink():entries.append({'path':rel,'type':'symlink','target':os.readlink(path),'mode':mode,'mtime_ns':s.st_mtime_ns});continue
  if path.is_dir():entries.append({'path':rel,'type':'directory','mode':mode,'mtime_ns':s.st_mtime_ns});continue
  if not path.is_file():excluded.append({'path':rel,'reason':'special_file'});continue
  reason=None
  if path.name in ['.env','.git-credentials','.netrc','.npmrc','.pypirc','id_rsa','id_ed25519']:reason='credential_filename'
  elif s.st_size<16*1024**2:
   b=path.read_bytes()
   if b'\x00' not in b[:4096] and PATTERN.search(b):reason='credential_marker'
  if reason:excluded.append({'path':rel,'reason':reason});continue
  entries.append({'path':rel,'type':'file','size':s.st_size,'mode':mode,'mtime_ns':s.st_mtime_ns})
 return entries,excluded

def fingerprint(entries):
 return hashlib.sha256(json.dumps(entries,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def repository_index(final=False):
 count=sum(v['type']=='file' for v in MANIFEST['entries']);total=sum(v.get('size',0) for v in MANIFEST['entries'])
 phase='COMPLETE — all assets verified' if final else 'IN PROGRESS — data release remains draft until verification completes'
 body=f'''# Full server experiment snapshot — {TAG}\n\nStatus: **{phase}**.\n\nSource: `/root/autodl-tmp/layer-probe` on the experiment server. Snapshot: {count:,} regular files, {total:,} logical bytes, plus recorded directory and symbolic-link metadata. Excluded credentials/special files: {len(MANIFEST['excluded'])}.\n\nData travel directly from the experiment server to GitHub Releases. The user's computer is not a data relay. Code, prompts, generated answers, raw results, activations, adapter checkpoints, model weights, and the existing project environment/cache are included. Original source files are not removed or modified.\n\nDownload the release assets and use `restore.py` to verify and reconstruct the directory. Large files are losslessly gzip-compressed in independent 512-MiB input parts; the manifest maps every part to its original path, offset and SHA256. Small files are stored in a separate gzip-compressed tar archive. Every uploaded asset is verified against GitHub's SHA256 (or read back on the server if unavailable).\n\nRelease: https://github.com/{REPO}/releases/tag/{TAG}\n\n```bash\npython3 restore.py --destination ./layer-probe-restored\n```\n\nThis is an archival experiment snapshot, not a new evaluation or license grant. Original model/data licenses continue to apply; recorded environment symlinks may require the corresponding system interpreter to exist.\n'''
 contents={'README.md':body,'restore.py':(WORK/'restore.py').read_text(),'uploader.py':Path(__file__).read_text()}
 if final:contents['VERIFIED.json']=json.dumps({'release_url':f'https://github.com/{REPO}/releases/tag/{TAG}','snapshot_fingerprint':MANIFEST['snapshot_fingerprint'],'logical_bytes':total,'regular_files':count,'manifest_asset':MANIFEST['manifest_asset'],'asset_count':len(MANIFEST['assets']),'all_verified':True},indent=2)+'\n'
 base=f'/repos/{REPO}';ref=api('GET',base+'/git/ref/heads/main');head=api('GET',base+'/git/commits/'+ref['object']['sha']);tree=[]
 for name,text in contents.items():
  blob=api('POST',base+'/git/blobs',{'content':text,'encoding':'utf-8'})
  tree.append({'path':f'archives/{TAG}/{name}','mode':'100644','type':'blob','sha':blob['sha']})
 built=api('POST',base+'/git/trees',{'base_tree':head['tree']['sha'],'tree':tree})
 commit=api('POST',base+'/git/commits',{'message':('Complete verified' if final else 'Prepare direct server')+' experiment data archive '+TAG,'tree':built['sha'],'parents':[ref['object']['sha']]})
 api('PATCH',base+'/git/refs/heads/main',{'sha':commit['sha'],'force':False})
 status(index_commit=commit['sha'])
 return commit['sha']

def record_asset(row):
 with GUARD:
  MANIFEST['assets'][row['name']]=row
  STATE['uploaded_assets']=len(MANIFEST['assets']);STATE['uploaded_compressed_bytes']=sum(v['size'] for v in MANIFEST['assets'].values())
  atomic(WORK/'manifest.json',MANIFEST);atomic(WORK/'status.json',STATE)

def consumer(jobs):
 while True:
  item=jobs.get()
  try:
   if item is None:return
   if HALT.is_set():continue
   path,name,relative,part=item
   row=send_asset(path,name)
   with GUARD:
    row.update(part)
    if relative=='__small__':MANIFEST['small_archive']=row
    else:MANIFEST['large_files'][relative]['parts'][str(part['part'])]=row
    record_asset(row)
    STATE['verified_input_bytes']=sum(x.get('raw_size',0) for x in MANIFEST['assets'].values())
    status('uploading',last_verified_asset=name)
   path.unlink()
   log('asset_verified',name=name,compressed_bytes=row['size'],verified_input_bytes=STATE['verified_input_bytes'],total_input_bytes=STATE['total_input_bytes'])
  except BaseException as e:
   HALT.set();status('failed',error=type(e).__name__+': '+str(e));log('upload_failed',error=type(e).__name__+': '+str(e))
  finally:jobs.task_done()

def worker(token):
 global TOKEN,RELEASE,ASSETS,MANIFEST
 TOKEN=token;resource.setrlimit(resource.RLIMIT_CORE,(0,0));WORK.mkdir(mode=0o700,parents=True,exist_ok=True);(WORK/'staging').mkdir(exist_ok=True)
 status('inventory',started_epoch=START,uploaded_assets=0,uploaded_compressed_bytes=0,verified_input_bytes=0)
 profile=api('GET','/user')
 if profile['login'].lower()!='liushengyu258':raise RuntimeError('Unexpected GitHub account')
 entries,excluded=snapshot();fp=fingerprint(entries)
 if (WORK/'manifest.json').exists():
  MANIFEST=json.loads((WORK/'manifest.json').read_text())
  if MANIFEST['snapshot_fingerprint']!=fp:raise RuntimeError('Source inventory changed since previous attempt; refusing mixed snapshot')
 else:
  MANIFEST={'format':'layer-probe-github-snapshot/1','source_root':str(ROOT),'repo':REPO,'tag':TAG,'created_epoch':time.time(),'snapshot_fingerprint':fp,'entries':entries,'excluded':excluded,'small_archive':None,'large_files':{},'assets':{}}
  persist()
 try:RELEASE=api('GET',f'/repos/{REPO}/releases/tags/{TAG}')
 except urllib.error.HTTPError as e:
  if e.code!=404:raise
  releases=api('GET',f'/repos/{REPO}/releases?per_page=100');RELEASE=next((x for x in releases if x['tag_name']==TAG),None)
  if not RELEASE:RELEASE=api('POST',f'/repos/{REPO}/releases',{'tag_name':TAG,'target_commitish':'main','name':'Full experiment server data — 2026-10-23','draft':True,'prerelease':False,'body':'Full project snapshot uploaded directly from the experiment server. Publication occurs only after all archive parts and the manifest have been verified.'})
 ASSETS=release_assets()
 entries=MANIFEST['entries']
 files=[v for v in entries if v['type']=='file'];small=[v for v in files if v['size']<SMALL];large=[v for v in files if v['size']>=SMALL]
 anticipated=1+sum(math.ceil(v['size']/CHUNK) for v in large)+3
 if anticipated>1000:raise RuntimeError('Snapshot exceeds release asset count limit')
 status('preparing',release_id=RELEASE['id'],release_url=f'https://github.com/{REPO}/releases/tag/{TAG}',regular_files=len(files),total_input_bytes=sum(v['size'] for v in files),expected_assets=anticipated,excluded_count=len(excluded))
 repository_index(False)
 jobs=queue.Queue(maxsize=2);threads=[threading.Thread(target=consumer,args=(jobs,),daemon=True) for _ in range(2)]
 for t in threads:t.start()
 smallpath=WORK/'staging/small-files.tar.gz'
 if MANIFEST.get('small_archive'):
  row=MANIFEST['small_archive'];remote=ASSETS.get(row['name'])
  if not remote or not confirm_asset(remote,row['size'],row['sha256']):raise RuntimeError('Prior small-file archive failed verification')
 else:
  with smallpath.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0,filename='') as out,tarfile.open(fileobj=out,mode='w|') as tar:
   for item in small:
    path=ROOT/item['path'];item['sha256']=digest_file(path);tar.add(path,arcname=item['path'],recursive=False)
  jobs.put((smallpath,'small-files.tar.gz','__small__',{'raw_size':sum(v['size'] for v in small)}));persist()
 # Irreplaceable experiment checkpoints first; activations next; downloadable base weights last.
 def priority(v):
  name=v['path'];return (0 if name.endswith('.pt') else 2 if name.startswith('models/') else 1,name)
 for item in sorted(large,key=priority):
  if HALT.is_set():break
  rel=item['path'];path=ROOT/rel
  with GUARD:
   info=MANIFEST['large_files'].setdefault(rel,{'size':item['size'],'parts':{},'sha256':None})
  expected=math.ceil(item['size']/CHUNK)
  if info.get('sha256') and len(info['parts'])==expected and all(p['name'] in ASSETS and ASSETS[p['name']].get('digest')=='sha256:'+p['sha256'] for p in info['parts'].values()):continue
  whole=hashlib.sha256();offset=0;number=0
  with path.open('rb') as src:
   while offset<item['size']:
    if HALT.is_set():break
    length=min(CHUNK,item['size']-offset);name='f-'+hashlib.sha256(rel.encode()).hexdigest()[:16]+f'-p{number:04d}.bin.gz';stage=WORK/'staging'/name
    previous=info['parts'].get(str(number))
    if previous and previous['name'] in ASSETS:
     if not confirm_asset(ASSETS[name],previous['size'],previous['sha256']):raise RuntimeError('Saved part failed remote verification')
     remaining=length
     while remaining:
      b=src.read(min(8*1024**2,remaining))
      if not b:raise RuntimeError('Source ended unexpectedly: '+rel)
      whole.update(b);remaining-=len(b)
    else:
     import shutil
     if shutil.disk_usage(WORK).free<3*1024**3:raise RuntimeError('Archive staging headroom below 3 GiB')
     part_hash=hashlib.sha256();remaining=length
     with stage.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0,filename='') as out:
      while remaining:
       b=src.read(min(8*1024**2,remaining))
       if not b:raise RuntimeError('Source ended unexpectedly: '+rel)
       whole.update(b);part_hash.update(b);out.write(b);remaining-=len(b)
     jobs.put((stage,name,rel,{'part':number,'offset':offset,'raw_size':length,'raw_sha256':part_hash.hexdigest()}))
    offset+=length;number+=1
  if HALT.is_set():break
  current=path.stat()
  if current.st_size!=item['size'] or current.st_mtime_ns!=item['mtime_ns']:raise RuntimeError('Source changed during snapshot: '+rel)
  with GUARD:info['sha256']=whole.hexdigest();item['sha256']=whole.hexdigest();persist()
 for _ in threads:jobs.put(None)
 jobs.join()
 if HALT.is_set():raise RuntimeError('One or more asset uploads failed; progress retained for resume')
 for item in large:
  info=MANIFEST['large_files'][item['path']]
  if len(info['parts'])!=math.ceil(item['size']/CHUNK) or not info['sha256']:raise RuntimeError('Incomplete source file: '+item['path'])
 status('verifying')
 # Ensure every original source entry still has the captured size/mtime.
 for item in files:
  s=(ROOT/item['path']).stat()
  if s.st_size!=item['size'] or s.st_mtime_ns!=item['mtime_ns']:raise RuntimeError('Source changed before completion: '+item['path'])
 MANIFEST['completed_epoch']=time.time();MANIFEST['complete']=True;persist()
 manpath=WORK/'staging/manifest.json.gz'
 with manpath.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0,filename='') as out:out.write(json.dumps(MANIFEST,ensure_ascii=False,indent=2).encode())
 manasset=send_asset(manpath,'manifest.json.gz');MANIFEST['manifest_asset']=manasset
 restore_asset=send_asset(WORK/'restore.py','restore.py')
 persist();commit=repository_index(True)
 body=f'Complete, checksum-verified project snapshot from the experiment server. {len(files):,} regular files, {sum(v["size"] for v in files):,} logical bytes, plus directories and symlinks. Assets uploaded directly server-to-GitHub. No source data were deleted.\n\nRun the attached restore.py to download, verify and reconstruct the original layout. Manifest: manifest.json.gz. Repository index: archives/{TAG}/README.md.\n\nCode/data/model licenses remain those of the original sources.'
 api('PATCH',f'/repos/{REPO}/releases/{RELEASE["id"]}',{'draft':False,'body':body,'target_commitish':commit})
 status('complete',completed_epoch=time.time(),release_url=f'https://github.com/{REPO}/releases/tag/{TAG}',index_commit=commit,manifest_asset_sha256=manasset['sha256'],verified_input_bytes=sum(v['size'] for v in files),all_verified=True)
 log('archive_complete',release_url=STATE['release_url'],files=len(files))

def launch():
 global TOKEN
 config=json.loads(sys.stdin.readline());token=config['token'];del config
 WORK.mkdir(mode=0o700,parents=True,exist_ok=True);os.chmod(WORK,0o700)
 lock=(WORK/'lock').open('a+')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print(json.dumps({'already_running':True}));return
 pid=os.fork()
 if pid:print(json.dumps({'started_pid':pid,'work_directory':str(WORK)}),flush=True);return
 os.setsid();os.umask(0o077)
 null=os.open('/dev/null',os.O_RDONLY);os.dup2(null,0);os.close(null)
 out=os.open(str(WORK/'worker.log'),os.O_CREAT|os.O_WRONLY|os.O_APPEND,0o600);os.dup2(out,1);os.dup2(out,2);os.close(out)
 try:worker(token)
 except BaseException as e:
  # Never log request headers, input credentials or an environment dump.
  message=type(e).__name__+': '+str(e)
  if token:message=message.replace(token,'[REDACTED]')
  status('failed',error=message);log('archive_failed',error=message)
 finally:
  TOKEN=None;token=None;lock.close()
 os._exit(0)

if __name__=='__main__':launch()
