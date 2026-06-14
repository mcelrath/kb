"""Probe the tree-sitter-rust API to understand node types."""
import tree_sitter_rust
from tree_sitter import Language, Parser

RUST = Language(tree_sitter_rust.language())
parser = Parser(RUST)
src = b"""
/// A simple function
pub async fn hello(x: u32, y: &str) -> u32 { x + 1 }

pub struct Foo {
    pub x: u32,
}

pub enum Bar { A, B(u32) }

pub trait Greet {
    fn greet(&self) -> String;
    fn greet_loud(&self) -> String { self.greet().to_uppercase() }
}

impl Foo {
    pub fn bar(&self) -> u32 { self.x }
    fn baz(&self) {}
}

impl Greet for Foo {
    fn greet(&self) -> String { format!("Hello {}", self.x) }
}

pub type Alias = u32;
pub const MAX: u32 = 42;
pub static GREETING: &str = "hi";
macro_rules! my_macro { () => {} }

mod inner {
    pub fn inner_fn() {}
}
"""

tree = parser.parse(src)
root = tree.root_node
lines = src.split(b"\n")

def show_node(node, indent=0):
    text_preview = src[node.start_byte:node.end_byte][:60].decode(errors='replace').replace('\n', '\\n')
    print(" " * indent + f"{node.type} [{node.start_point[0]+1}:{node.start_point[1]}-{node.end_point[0]+1}:{node.end_point[1]}]  {text_preview!r}")
    if indent < 4:
        for child in node.named_children:
            show_node(child, indent + 2)

show_node(root)
