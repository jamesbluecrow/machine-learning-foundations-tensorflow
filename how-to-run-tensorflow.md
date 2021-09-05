https://www.tensorflow.org/install/docker

Run console:

```
docker run -it tensorflow/tensorflow bash
```

Run on CPU
```
docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./script.py
```

### Alternatively use tensorflow-macos which is a bit faster

When installing from ide, go to:
`preferences / interpreter / + / tensorflow-macos`