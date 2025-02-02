## TODO

### Paper
- [x] 生成gist的perf
- [x] 重构batch搜索接口，传参统一使用std::array
- [x] 实现batch8
- [x] 去除不必要的Prefetch（当前访问邻居时即使它已经访问过也会Prefetch）
- [x] GloVE
- [x] 实现HNSW++算法
- [x] 解决NYTimes和Glove数据集高recall性能劣化严重的问题
- [x] Save Cleaned results Data
- [ ] OPQ
- [ ] Refactor OPQ
- [ ] Use Google naming conventions
- [ ] HNSW range search
- [ ] IVFPQ ANN search
- [ ] IVFPQ range search
- [ ] 多线程搜索实现与benchmark
- [ ] Perf with VTune

### Project
- [ ] Use static setter map for SetterProxy
- [ ] Split declarations and definations of unity searcher
- [ ] IGetter, ISetter, GetterProxy, SetterProxy
- [ ] Unified graph index format
- [ ] Seperated storage of graph index and data
- [ ] Support build indexes with UNITY directly
- [ ] Support PQ FastScan
