extern crate proc_macro;

use proc_macro::TokenStream;
use std::collections::BTreeMap;
use proc_macro2::{Ident, Span};

use syn::{parse_macro_input, Token, token, Visibility, braced, parse::{Parse, ParseStream}, parenthesized};
use quote::quote;

#[derive(Clone, Debug)]
enum FreeGeneric {
    Type(Ident, bool),
    Const(Ident, Ident),
}

impl Parse for FreeGeneric {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.peek(Token![const]) {
            input.parse::<Token![const]>()?;
            let a = input.parse()?;
            input.parse::<Token![:]>()?;
            let b = input.parse()?;
            Ok(Self::Const(a, b))
        } else {
            Ok(if input.peek(token::Paren) {
                let ty;
                parenthesized!(ty in input);
                Self::Type(ty.parse()?, true)
            } else {
                Self::Type(input.parse()?, false)
            })
        }
    }
}

struct BuilderInfo {
    visibility: Visibility,
    name: Ident,
    free_generics: Vec<FreeGeneric>,
    fields: Vec<(Ident, Ident)>,

}

impl BuilderInfo {
    fn new(visibility: Visibility, name: Ident, free_generics: Vec<FreeGeneric>, fields: Vec<(Ident, Ident)>) -> Result<Self, ()> {
        Ok(Self {
            visibility,
            name,
            free_generics,
            fields,
        })
    }

    fn build(self) -> TokenStream {
        //! ```text
        //! struct Builder<fields, free_generics_with_wrapped_const>{...}
        //!
        //! impl<non_field_non_const> Builder<Uninitialized,...,non_const_free_generics> {
        //!     fn new<default_free_generics>() -> Builder<Uninitialized,...,free_generics_with_wrapped_const> {
        //!         Builder {
        //!             field: Uninitialized,
        //!             ...,
        //!             _marker: PhantomData<non_const_free_generics>
        //!         }
        //!     }
        //! }
        //!
        //!     impl<fields\[i], free_generics_with_wrapped_const> Builder<fields,...,Uninitialized,...,fields,free_generics_with_wrapped_const> {
        //!         fn [i]<[i]>(self, [i]: [i]) -> Builder<fields,free_generics_with_wrapped_const> {
        //!             let Self {
        //!                 fields,
        //!                 ...,
        //!                 [i]: _,
        //!                 ...,
        //!                 fields,
        //!                 _marker,
        //!             } = self;
        //!             Builder {
        //!                 fields,
        //!                 _marker,
        //!             }
        //!         }
        //!     }
        //! ```

        let Self { visibility, name, free_generics, fields } = self;

        let containers = free_generics.iter().cloned().filter_map(|g| match g {
            FreeGeneric::Type(_, _) => None,
            FreeGeneric::Const(_, t) => {
                let i = t.to_string() + "Container";
                Some((t, Ident::new(&i, Span::call_site())))
            }
        }).collect::<BTreeMap<_, _>>();

        let field_types = || fields.iter().cloned().map(|(_, t)| t);
        let fields0 = field_types();
        let full_fields = || fields.iter().cloned().map(|(a, b)| quote! {#a: #b});
        let fields1 = full_fields();
        let fields2 = fields.iter().map(|_| quote! {Uninitialized});
        let fields3 = fields.iter().map(|_| quote! {Uninitialized});
        let fields_names = || fields.iter().cloned().map(|(t, _)| t);
        let fields4 = fields_names();

        let free_generics0 = free_generics.iter().cloned().map(|g| match g {
            FreeGeneric::Type(t, true) => quote! {#t = Uninitialized},
            FreeGeneric::Type(t, false) => quote! {#t},
            FreeGeneric::Const(t, _) => quote! {#t = Uninitialized},
        });
        let free_generics1 = free_generics.iter().cloned().map(|g| match g {
            FreeGeneric::Type(t, _) | FreeGeneric::Const(t, _) => t,
        });
        let free_generics2 = free_generics.iter().cloned().map(|g| match g {
            FreeGeneric::Type(t, _) => quote! {#t},
            FreeGeneric::Const(a, b) => {
                let t = containers.get(&b).unwrap();
                quote! {#t<#a>}
            }
        });

        let non_default0 = free_generics.iter().cloned().filter_map(|g| match g {
            FreeGeneric::Type(t, false) => Some(t),
            _ => None
        });
        let non_default1 = free_generics.iter().cloned().filter_map(|g| match g {
            FreeGeneric::Type(t, false) => Some(t),
            _ => None
        });
        let default0 = free_generics.iter().cloned().filter_map(|g| match g {
            FreeGeneric::Type(t, true) => Some(quote! {#t}),
            FreeGeneric::Const(a, b) => Some(quote! {const #a: #b}),
            _ => None,
        });


        let fns = (0..fields.len()).map(|i| {
            let fields0 = field_types().enumerate().filter_map(|(j, t)| if j == i { None } else { Some(t) });
            let fields1 = field_types().enumerate().map(|(j, t)| if i == j { quote! {Uninitialized} } else { quote! {#t} });
            let fields2 = field_types();
            let (field_name, field_type) = fields[i].clone();
            let field_names0 = fields_names().enumerate().filter_map(|(j, t)| if j == i { None } else { Some(t) });
            let field_names1 = fields_names();

            let free_generics0 = free_generics.iter().cloned().map(|g| match g {
                FreeGeneric::Type(t, _) => quote! {#t},
                FreeGeneric::Const(a, b) => quote! {const #a: #b},
            });
            let free_generics1 = free_generics.iter().cloned().map(|g| match g {
                FreeGeneric::Type(t, _) => quote! {#t},
                FreeGeneric::Const(a, b) => {
                    let t = containers.get(&b).unwrap();
                    quote! {#t<#a>}
                }
            });
            let free_generics2 = free_generics.iter().cloned().map(|g| match g {
                FreeGeneric::Type(t, _) => quote! {#t},
                FreeGeneric::Const(a, b) => {
                    let t = containers.get(&b).unwrap();
                    quote! {#t<#a>}
                }
            });

            quote! {
                impl<#(#fields0,)* #(#free_generics0,)*> #name<#(#fields1,)* #(#free_generics1,)*> {
                    #visibility fn #field_name<#field_type>(self, #field_name: #field_type) -> #name<#(#fields2,)* #(#free_generics2,)*> { // todo: make const
                        let Self {
                            #(#field_names0,)*
                            #field_name: _,
                            _marker,
                        } = self;
                        Builder {
                            #(#field_names1,)*
                            _marker,
                        }
                    }
                }
            }
        });

        let containers = containers.iter().map(|(a, b)| quote! {#visibility struct #b<const T: #a>;});

        (quote! {
            #(#containers)*

            #visibility struct #name<#(#fields0,)* #(#free_generics0,)*> {
                #(#fields1,)*
                _marker: std::marker::PhantomData<(#(#free_generics1,)*)>
            }

            impl<#(#non_default0,)*> #name<#(#fields2,)* #(#non_default1,)*> {
                #visibility const fn new<#(#default0,)*>() -> #name<#(#fields3,)* #(#free_generics2,)*> {
                    #name {
                        #(#fields4: Uninitialized,)*
                        _marker: std::marker::PhantomData,
                    }
                }
            }

            #(#fns)*
        }).into()
    }
}

impl Parse for BuilderInfo {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let visibility = input.parse()?;

        input.parse::<Token![struct]>()?;

        let name = input.parse()?;

        let free_generics = if !input.peek(Token![<]) {
            vec![]
        } else {
            input.parse::<Token![<]>()?;
            let mut t = vec![input.parse()?];
            loop {
                if input.peek(Token![>]) { break; }
                input.parse::<Token![,]>()?;
                if input.peek(Token![>]) { break; }
                t.push(input.parse()?)
            }
            input.parse::<Token![>]>()?;
            t
        };

        let fields = {
            let fields;
            braced!(fields in input);
            fields
        }.parse_terminated::<_, Token![,]>(|input| {
            let a = input.parse()?;
            input.parse::<Token![:]>()?;
            let b = input.parse()?;
            Ok((a, b))
        })?.into_iter().collect();


        Ok(Self::new(visibility, name, free_generics, fields).map_err(|()| syn::Error::new(Span::call_site(), "error"))?)
    }
}

#[proc_macro]
pub fn builder(input: TokenStream) -> TokenStream {
    parse_macro_input!(input as BuilderInfo).build() // todo: add field defaults and constraints
}