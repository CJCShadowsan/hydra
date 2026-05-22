import XCTest
@testable import MeshLLM

func makeOwnerKeypairBytesHex() -> String {
    generateOwnerKeypairHex()
}

func makeTestNode() throws -> Node {
    try Node(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
}
