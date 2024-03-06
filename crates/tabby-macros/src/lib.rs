use proc_macro::TokenStream;
use quote::quote;
use syn::{bracketed, parse::Parse, parse_macro_input, Ident, LitStr, Token, Type};

struct Column {
    name: LitStr,
    non_null: bool,
}

impl Parse for Column {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name = input.parse()?;
        let non_null = input.peek(Token![!]);
        if non_null {
            input.parse::<Token![!]>()?;
        }
        Ok(Column { name, non_null })
    }
}

struct PaginationQueryInput {
    pub typ: Type,
    pub table_name: LitStr,
    pub columns: Vec<Column>,
    pub condition: Option<LitStr>,
    pub limit: Ident,
    pub skip_id: Ident,
    pub backwards: Ident,
}

mod kw {
    use syn::custom_keyword;

    custom_keyword!(FROM);
    custom_keyword!(WHERE);
}

impl Parse for PaginationQueryInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let typ = input.parse()?;
        input.parse::<Token![,]>()?;
        let table_name = input.parse()?;
        input.parse::<Token![,]>()?;
        let mut columns = vec![];

        let inner;
        bracketed!(inner in input);

        columns.push(inner.parse()?);

        while inner.peek(Token![,]) {
            inner.parse::<Token![,]>()?;
            columns.push(inner.parse()?);
        }

        let mut condition = None;
        if input.peek(kw::WHERE) {
            input.parse::<kw::WHERE>()?;
            condition = Some(input.parse()?);
        }

        input.parse::<Token![,]>()?;
        let limit = input.parse()?;
        input.parse::<Token![,]>()?;
        let skip_id = input.parse()?;
        input.parse::<Token![,]>()?;
        let backwards = input.parse()?;

        Ok(PaginationQueryInput {
            typ,
            table_name,
            columns,
            condition,
            limit,
            skip_id,
            backwards,
        })
    }
}

#[proc_macro]
pub fn query_paged_as(input: TokenStream) -> TokenStream {
    let input: PaginationQueryInput = parse_macro_input!(input);

    let typ = input.typ;
    let table_name = input.table_name;
    let columns = input
        .columns
        .into_iter()
        .map(|col| {
            let name = col.name.value();
            if col.non_null {
                format!("{name} as \"{name}!\"")
            } else {
                name
            }
        })
        .collect::<Vec<_>>();
    let columns_joined = columns.join(", ");
    let where_clause = input
        .condition
        .clone()
        .map(|cond| format!("WHERE {}", cond.value()))
        .unwrap_or_default();
    let condition = match input.condition {
        Some(cond) => quote! {Some(#cond.into())},
        None => quote! {None},
    };

    let limit = input.limit;
    let skip_id = input.skip_id;
    let backwards = input.backwards;
    quote! {
        {
            let _ = sqlx::query_as!(#typ, "SELECT " + #columns_joined + " FROM " + #table_name + #where_clause);
            crate::make_pagination_query_with_condition(#table_name, &[ #(#columns),* ], #limit, #skip_id, #backwards, #condition)
        }
    }.into()
}
