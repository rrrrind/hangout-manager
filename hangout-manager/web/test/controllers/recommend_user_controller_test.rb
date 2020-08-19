require 'test_helper'

class RecommendUserControllerTest < ActionDispatch::IntegrationTest
  test "should get new" do
    get recommend_user_new_url
    assert_response :success
  end

  test "should get create" do
    get recommend_user_create_url
    assert_response :success
  end

end
