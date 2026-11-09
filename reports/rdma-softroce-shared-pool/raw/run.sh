set -u
U="-e UCX_TLS=rc_v,self -e UCX_NET_DEVICES=rxe0:1"
D="--network host --device /dev/infiniband --ulimit memlock=-1 -e LIFEBOAT_POOL_TOKEN=t0k"
IMG=lbc-nixl-rdma:3
docker rm -f lbcpool-srv >/dev/null 2>&1
docker run -d --name lbcpool-srv $D $U $IMG lifeboat-cache-pool --size 8GB --listen POOL:7470 --transport nixl >/dev/null
sleep 6
docker logs lbcpool-srv 2>&1 | grep "lifeboat-cache-pool:" > pool.log
for mb in 0.5 2 8; do
  n=$(python3 -c "print(int(2048/$mb))")
  for tr in nixl tcp; do for r in 1 2 3; do
    docker run --rm $D $U -e UCX_PROTO_INFO=y -v $PWD:/out $IMG python benchmarks/pool_bench.py --server POOL:7470 --transport $tr --blocks $n --block-mb $mb --threads 4 --json /out/${tr}_${mb}MB_r$r.json > ${tr}_${mb}MB_r$r.log 2>&1 || echo FAIL $tr $mb $r
  done; done
done
docker rm -f lbcpool-srv >/dev/null
echo done
