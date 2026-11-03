"""Plan or explicitly restore omitted artifacts; never trains or contacts Runpod."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def safe_relative(value):
    path=PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe artifact path')
    return Path(*path.parts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--plan',action='store_true')
    mode.add_argument('--fetch',action='store_true',help='Explicitly fetch public URLs and verify hashes')
    mode.add_argument('--source-root',type=Path,help='Copy verified files from an existing local workspace')
    parser.add_argument('--meshes-only',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'artifacts/restore_manifest.json').read_text())
    entries=manifest['meshes']+([] if args.meshes_only else manifest['models'])
    if not args.fetch and args.source_root is None:
        print(json.dumps({'plan_only':True,'artifacts':entries},indent=2));return
    for entry in entries:
        relative=safe_relative(entry['path']);target=ROOT/relative
        if not target.resolve().is_relative_to(ROOT):raise ValueError('Artifact escapes checkout')
        if target.exists():
            if digest(target)!=entry['sha256']:raise ValueError(f'Existing artifact differs: {relative}')
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix(target.suffix+'.partial')
        if temporary.exists():raise ValueError(f'Existing partial artifact requires inspection: {relative}')
        if args.source_root is not None:
            source=args.source_root.resolve()/relative
            if not source.resolve().is_relative_to(args.source_root.resolve()):raise ValueError('Source escapes workspace')
            shutil.copyfile(source,temporary)
        else:
            url=entry.get('url')
            if not url:raise ValueError(f'No pinned public URL recorded; use --source-root for {relative}')
            parsed=urlparse(url)
            if parsed.scheme!='https' or parsed.hostname not in ('raw.githubusercontent.com','huggingface.co'):
                raise ValueError('Only recorded public upstream HTTPS sources are allowed')
            with urlopen(url,timeout=60) as response,temporary.open('xb') as stream:
                total=0
                for block in iter(lambda:response.read(1024*1024),b''):
                    total+=len(block)
                    if total>entry['bytes']:raise ValueError('Download exceeds expected artifact size')
                    stream.write(block)
        if temporary.stat().st_size!=entry['bytes'] or digest(temporary)!=entry['sha256']:
            raise ValueError(f'Artifact failed size/hash verification: {relative}; partial retained')
        temporary.replace(target)
        print(f'Restored and verified {relative}')


if __name__=='__main__':main()
