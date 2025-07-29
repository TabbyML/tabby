use proc_macro::TokenStream;
use quote::quote;
use syn::{bracketed, parse::Parse, parse_macro_input, Expr, Ident, LitStr, Token, Type};

#[derive(Clone)]
struct Column {
    name: LitStr,
    non_null: bool,
    rename: LitStr,
}

impl Parse for Column {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name: LitStr = input.parse()?;
        let non_null = input.peek(Token![!]);
        if non_null {
            input.parse::<Token![!]>()?;
        }
        let mut rename = None;
        if input.peek(Token![as]) {
            input.parse::<Token![as]>()?;
            rename = Some(input.parse()?);
        }
        Ok(Column {
            rename: rename.unwrap_or(name.clone()),
            name,
            non_null,
        })
    }
}

struct PaginationQueryInput {
    pub typ: Type,
    pub table_name: LitStr,
    pub columns: Vec<Column>,
    pub condition: Option<Expr>,
    pub limit: Ident,
    pub skip_id: Ident,
    pub backwards: Ident,
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

        input.parse::<Token![,]>()?;
        let limit = input.parse()?;
        input.parse::<Token![,]>()?;
        let skip_id = input.parse()?;
        input.parse::<Token![,]>()?;
        let backwards = input.parse()?;

        let mut condition = None;
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            condition = Some(input.parse()?);
        }

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
        .iter()
        .map(|col| {
            let name = col.name.value();
            let rename = col.rename.value();
            let non_null = if col.non_null {
                "!"
            } else {
                Default::default()
            };
            format!("{name} AS '{rename}{non_null}'")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let column_args: Vec<String> = input.columns.iter().map(|col| col.name.value()).collect();
    let limit = input.limit;
    let condition = match input.condition {
        Some(cond) => quote! {#cond},
        None => quote! {None},
    };
    let skip_id = input.skip_id;
    let backwards = input.backwards;
    quote! {
        sqlx::query_as(&crate::make_pagination_query_with_condition({
            let _ = sqlx::query_as!(#typ, "SELECT " + #columns + " FROM (SELECT * FROM " + #table_name + ")");
            &#table_name
        }, &[ #(#column_args),* ], #limit, #skip_id, #backwards, #condition))
    }
    .into()
}
