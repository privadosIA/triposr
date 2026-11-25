# TripoSR.Net

C# port (.NET 8 + TorchSharp) of TripoSR, inference only: one picture in, one binary glTF out. It replaces the Python container of Veritmo as the product's engine for these models. **Status:** the whole model is ported and every stage matches the reference (see the porting plan); what remains is the publish script and the `ServiceSpec` in Veritmo.

## Articles

- [Porting plan](porting-plan.md): the model stage by stage, what each stage is in the reference and where it goes here, and how it is checked.
- [The HTTP service](http-service.md): the contract, the same as the Python container's.

## API reference

The public API of the `TripoSR` library, generated from the XML comments of the source.
