def get_multires_hash_encoding(args, encoder = HashEncoder):
    '''
    Returns a multiresolutional hash encoding and output dimension.
    '''
    
    embedded = encoder(bounding_box=args.bounding_box, \
                        log2_hashmap_size=args.log2_hashmap_size, \
                        finest_resolution=args.finest_resolution)

    out_dim = embedded.out_dim
    
    return embedded, out_dim
    
    
    