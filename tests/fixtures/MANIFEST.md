# Fixtures of the golden run

Made by `tools/golden_all.sh` (the Python reference, TripoSR commit `107cefd`, inside `mediaorquestator/model3d:1`, fp32 on the CPU) with
`tools/dump_reference.py`; the weights are not here and never are. Pictures are examples that come with TripoSR (MIT licence).

What is committed: the pictures, the cut-outs and inputs (`*.png`), the per-picture summaries (`*.json`), `ckpt_keys.json` and `versions.json`.
What is regenerated (too big for the repository, ignored by git): `*.safetensors` (stages, shared tensors, raw meshes) and `*.glb`.
The hash of a regenerated file can differ from this table between machines (the order of float sums changes with the threads and the
CPU): the tests load every fixture through `FixtureDirectory`, which checks its sha256 against this table and fails naming the file when they
differ, so after a deliberate golden run on another machine rewrite the table with `tools/write_manifest.sh`. `tools/verify_manifest.sh` checks
the committed files strictly and the regenerated ones only for being there, with the size.

| file | bytes | sha256 | committed |
|---|---:|---|:---:|
| `chair.128.mesh.safetensors` | 777256 | `ff3ffa094c46676b9f2b8cf5f171389c0ddf6dcb35003d90847e6204c669cd1e` | no |
| `chair.192.mesh.safetensors` | 1775536 | `56778458a3d4eba6ff5387671b727e8645ac31cf60c6b8859fe5706dc6da0a20` | no |
| `chair.256.mesh.safetensors` | 3181992 | `5be16738b752c6118cdce1bc4a339c810abb59d04a461da318eba4b82c511e6c` | no |
| `chair.cutout.png` | 114705 | `2503c12a74419d91a4c6c9f1affc48fee6e2b8b9091956ca6211e91ada57b5bf` | yes |
| `chair.high.glb` | 1675584 | `898a30e8b192e7992edbf7dc58fe4cdbe10dd34a2439dd2d82993089ce3a8922` | no |
| `chair.input.png` | 93377 | `ce82549c713578b6a21fa4fb0f2c4756131b0296f42bdc4fbd1a3ba220800c37` | yes |
| `chair.json` | 1191 | `a40cd40a103850af1ab87a61d83f22ea6f0a2cc648bc85e719475fa335d93f87` | yes |
| `chair.low.glb` | 409928 | `a38d298835832cea145efba36bdbc91ae91f6cb374e46bb9b8eadd43216bb745` | no |
| `chair.normal.glb` | 935340 | `4d47202a1caf13b0c73abb054dd1bcca30e129c88b964c2e2e9715c279d8d3aa` | no |
| `chair.stages.safetensors` | 29314736 | `5f76b9f10798f91e536e2e86bdad6ba23fdeae76b11fba1086566bd1fb6ec0e8` | no |
| `ckpt_keys.json` | 111216 | `3e2d2a2703ee2b6f0aba1c18b6d57cbc2497a6a22891911b0b0f415ba01faed9` | yes |
| `hamburger.128.mesh.safetensors` | 1009452 | `4c2b45b33fd57c1b6b1e5e82fb52289b7e985b2f6ab1b60fe96de7a4d468d0e8` | no |
| `hamburger.192.mesh.safetensors` | 2371676 | `d91a97a3a1460c21dd4fcdf045734da3e1fa28d085fc0cc191d7fe14e298d0b3` | no |
| `hamburger.256.mesh.safetensors` | 4211156 | `be330b18e69527b642417fd039743ac21228e5eb6a9fdd5c5b864cbb2444b380` | no |
| `hamburger.cutout.png` | 125172 | `2f16cb1f2bb932ae643cd4acc09bb6ecb51c84eca32dd5a7c71ea4980b40b51a` | yes |
| `hamburger.high.glb` | 2217264 | `ccc3f81bb42e3df0fccb8811b5727de325cf3cc7aa3bad785425340f9bcaf6b2` | no |
| `hamburger.input.png` | 106903 | `1b27258db875911505959656e98fb057ec89195fe99eb88650f48b0d43b14b85` | yes |
| `hamburger.json` | 1197 | `158d52f23cfc16f14823463c6749e30324e5dd2823c09aeb7341059de21ee2f6` | yes |
| `hamburger.low.glb` | 532144 | `b2e8f97925659710a8e8ba46bfed66a8080b5768fa1a44109b08649d83ed3326` | no |
| `hamburger.normal.glb` | 1249112 | `7e67e251bdd260202b72fb77cc265b8317da522a4a2771db19c126f3d8d1fb31` | no |
| `hamburger.stages.safetensors` | 29314736 | `2780d75145fe4f57b62505abf65b445597ebd316d452f40be3f4ab6d3f2e1970` | no |
| `horse.128.mesh.safetensors` | 814172 | `8e25c2ea576e812d4c8e62e31f6bbebe5ce0e99bb11af7df42fb06b2a4cd693e` | no |
| `horse.192.mesh.safetensors` | 1936652 | `2ca54bf527f93dc399b2b93ba004bf551dfa7138836f2a084a5d00717d9445db` | no |
| `horse.256.mesh.safetensors` | 3420340 | `ac88dff443e922c0461c50ec36cbdd2c93662de9b5853ff267b45333be498a6a` | no |
| `horse.cutout.png` | 242273 | `05bd035f7e5c10390912ffef1012443d16d743557aceee6f5c39afa9e977c6e1` | yes |
| `horse.high.glb` | 1801048 | `e43623d5ee75dcc3e40537c702fbd4b334af87c38f47e45301915c4647e556e0` | no |
| `horse.input.png` | 200249 | `ba2a6ca0033b35ba2bc4025ffffa50f61ba5a8266c8b22d7c02ca716c6eca97f` | yes |
| `horse.json` | 1190 | `f3d1f37af015c8e42baf74b663dfecd8860cb7da5b98322ba9b309d877774e38` | yes |
| `horse.low.glb` | 429360 | `15af6d3178404488db48c39b92d3c5ab17eec7252065089e64c3c43d60e648ca` | no |
| `horse.normal.glb` | 1020172 | `6b0435e44bac2010baabeba32a5084f6c9834aba82a498b6823497c92ef433ab` | no |
| `horse.stages.safetensors` | 29314736 | `f06231136fddff828fb14bf76c46d00f02073366addd21433e84c4f58b88abd8` | no |
| `pictures/chair.png` | 114705 | `2503c12a74419d91a4c6c9f1affc48fee6e2b8b9091956ca6211e91ada57b5bf` | yes |
| `pictures/hamburger.png` | 125172 | `2f16cb1f2bb932ae643cd4acc09bb6ecb51c84eca32dd5a7c71ea4980b40b51a` | yes |
| `pictures/horse.png` | 257687 | `629a0cc78a18e16f359afb22e2d8b9594f435ae09326560c86b0b095650d54e0` | yes |
| `pictures/poly_fox.png` | 633575 | `255b3fe96ec357d8744343849f8d7cb39156988520c7ca0264edc6689d6e31ca` | yes |
| `pictures/teapot.png` | 275764 | `66b0b8f7dd759e97d20665999885e0548f96c95dadee6bf9762ccd877c596b8c` | yes |
| `poly_fox.128.mesh.safetensors` | 1047264 | `c9a0a6bfda51f3ecd9ac643a8ec0e34c46a35adaed81c102c9776bd033f71fd8` | no |
| `poly_fox.192.mesh.safetensors` | 2421744 | `7e4bded32ff30fb2ed53ad7eff9f744db3c58cc765d74f0a7a8fb0a74a7ce53c` | no |
| `poly_fox.256.mesh.safetensors` | 4337728 | `af8ba96b6331b6811267552738bcdbef872b99801dcc11cde5699eeae715a6a1` | no |
| `poly_fox.cutout.png` | 756741 | `57d43ad362546ddcef020d6e1f271e1b99fb8106f6e2b8cbcf9a43db9a34245f` | yes |
| `poly_fox.high.glb` | 2283872 | `e6e33442eaeab68518d204ff901984d63bc0a99377d7424e76d1af126a92dd0b` | no |
| `poly_fox.input.png` | 629596 | `4e38097baf66ef6769ebfc4f96e5dfa3e7d7cdb597a418eb09d5cd7a09ff153a` | yes |
| `poly_fox.json` | 1195 | `ca100ae520247a345298fe20c8565b5207b62e8dcfe9d7f906e5ce9772868f35` | yes |
| `poly_fox.low.glb` | 552040 | `9b18fe1385e320aacd9fcc3b33429435706c8505c9f5dae4bed8d9aaac68532f` | no |
| `poly_fox.normal.glb` | 1275452 | `a34798c3ca2f3b88a1a760e4f04fd8b1a807c07c03c33cfa72452d6acd7f35a7` | no |
| `poly_fox.stages.safetensors` | 29314736 | `5dcbeec7a675b5ab23c3f4f7c581b52349b7392b1a62f7cdd995e8f640bdfcb6` | no |
| `shared.safetensors` | 12632232 | `f92bb689963b69117db469e6e2a66029986a1b39a9044408e424c956693da8d4` | no |
| `teapot.128.mesh.safetensors` | 771348 | `690f4a968f058e8c0cdcfc246fa661ac13bed855646c925092698469ddd971c9` | no |
| `teapot.192.mesh.safetensors` | 1773160 | `d21144b40d66a411663ac063d575099816d332b84dfec0d27225eed15224d9d4` | no |
| `teapot.256.mesh.safetensors` | 3162952 | `37d71a29ac24b53774991b2cd3cbe79d1fdab777a20e72b430aacfdf705e47d0` | no |
| `teapot.cutout.png` | 338089 | `59c53a685d391f25a1ea3cefa297e4391f44965a73b50bca42ef2834dc2f6b02` | yes |
| `teapot.high.glb` | 1665568 | `49d05541cc1e9061ad108eda46137f3e7c12d7a9869ea1b97aaedc33a7c4682e` | no |
| `teapot.input.png` | 273830 | `a83872d5ad8cc81a9cda5cd95a1dcc5f1eef87cc1b029c569a00127a47ade232` | yes |
| `teapot.json` | 1196 | `46037707e5738f811d23f0bdf0547b5f72e627dd86c5a80b5bafdb07c62d1379` | yes |
| `teapot.low.glb` | 406820 | `e3f5ea20f7459520e4d33eae41038b7ec22a058d2d0187ea3dd6df3539b75cef` | no |
| `teapot.normal.glb` | 934092 | `d02d4490ead192420bef5c9deb7f3f4bda9ce521829275c9abe6571df0389062` | no |
| `teapot.stages.safetensors` | 29314736 | `bf4e00231b4e72deed946ec66bdf83bc03deaecbeb16ceaada663390ac6f904f` | no |
| `versions.json` | 319 | `b343216caa00e69ac4a8d9724b692dec9c11a060b520338c51c47d28668055c8` | yes |
